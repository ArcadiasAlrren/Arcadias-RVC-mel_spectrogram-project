import torch
import torch.nn as nn
import torch.nn.functional as F

class RVCModel(nn.Module):
    def __init__(self, config):
        super(RVCModel, self).__init__()
        self.hidden_size = config["model"]["hidden_size"]
        self.embedding_size = config["model"]["embedding_size"]
        self.num_layers = config["model"]["num_layers"]
        self.n_mels = config["data"]["n_mels"]
        
        self.encoder = nn.LSTM(self.n_mels, self.hidden_size, self.num_layers, batch_first=True)
        self.decoder = nn.LSTM(self.hidden_size, self.n_mels, self.num_layers, batch_first=True)
    
    def forward(self, mel_spec, hidden):
        output, hidden = self.encoder(mel_spec, hidden)
        mel_output, _ = self.decoder(output, hidden)
        return mel_output, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)