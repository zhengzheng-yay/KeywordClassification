import torch
import torch.nn as nn
from torchaudio import transforms

class logFbankCal(nn.Module):                                                   
    def __init__(self, feats_config):
        super(logFbankCal, self).__init__()
        self.sample_rate = int(feats_config['sample_rate'])
        self.n_fft = int(feats_config['n_fft'])
        self.win_length = int(feats_config['win_length'] * self.sample_rate)
        self.hop_length = int(feats_config['win_shift'] * self.sample_rate)
        self.n_mels = int(feats_config['n_mels'])
        self.fbankCal = transforms.MelSpectrogram(sample_rate=self.sample_rate,      
                                                  n_fft=self.n_fft,                  
                                                  win_length=self.win_length,        
                                                  hop_length=self.hop_length,        
                                                  n_mels=self.n_mels)                
                                                                                
    def forward(self, x):                                            
        out = self.fbankCal(x)                                                                                                
        out = torch.log(out + 1e-6)                                             
        out = out - out.mean(axis=1).unsqueeze(dim=1)
        out = out.transpose(1,2)                                                                           
        return out