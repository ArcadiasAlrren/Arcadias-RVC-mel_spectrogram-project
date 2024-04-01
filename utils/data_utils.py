import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class VoiceDataset(Dataset):
    def __init__(self, data_dir, sample_rate, n_mels, n_fft, hop_length):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.audio_files = sorted(os.listdir(data_dir))
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index):
        audio_path = os.path.join(self.data_dir, self.audio_files[index])
        
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=self.n_mels,
                                                  n_fft=self.n_fft, hop_length=self.hop_length)
        mel_spec = torch.from_numpy(mel_spec).float()
        
        return mel_spec