import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Сортировка батча по длине последовательностей (для упаковки)
    batch.sort(key=lambda x: x.shape[1], reverse=True)
    
    # Упаковка последовательностей разной длины
    seq_lengths = [seq.shape[1] for seq in batch]
    padded_seqs = pad_sequence(batch, batch_first=True)
    
    return padded_seqs, seq_lengths

def train_epoch(model, dataloader, criterion, optimizer, device, tbptt_steps=10):
    model.train()
    running_loss = 0.0
    hidden = None
    
    for mel_spec, seq_lengths in dataloader:
        mel_spec = mel_spec.to(device)
        
        if hidden is None:
            hidden = model.init_hidden(mel_spec.size(0), device)
        
        for i in range(0, mel_spec.size(1), tbptt_steps):
            mel_spec_chunk = mel_spec[:, i:i+tbptt_steps]
            
            optimizer.zero_grad()
            
            mel_spec_chunk = mel_spec_chunk.transpose(1, 2)
            mel_output, hidden = model(mel_spec_chunk, hidden)
            mel_output = mel_output.transpose(1, 2)
            
            mel_spec_chunk = mel_spec_chunk.transpose(1, 2).contiguous().view(-1, model.n_mels)
            mel_output = mel_output.contiguous().view(-1, model.n_mels)
            
            loss = criterion(mel_output, mel_spec_chunk)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * mel_spec_chunk.size(0)
        
        # Отсоединение скрытых состояний от графа вычислений
        hidden = (hidden[0].detach(), hidden[1].detach())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for mel_spec, seq_lengths in dataloader:
            mel_spec = mel_spec.to(device)
            seq_lengths = torch.tensor(seq_lengths).to(device)  # Преобразование в тензор и перенос на устройство
            
            # Передача мел-спектрограмм и длин последовательностей в модель
            mel_output = model(mel_spec, seq_lengths)
            
            # Обрезка целевых мел-спектрограмм до исходной длины последовательностей
            mel_spec = [mel_spec[i, :seq_lengths[i]] for i in range(mel_spec.size(0))]
            mel_spec = torch.cat(mel_spec, dim=0)
            
            loss = criterion(mel_output, mel_spec)
            
            running_loss += loss.item() * mel_spec.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss