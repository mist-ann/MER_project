"""
Fixed LSTM Model - bez sigmoid w output
Używa L1Loss zamiast MSELoss dla lepszej convergence
"""

import torch
import torch.nn as nn


class ImprovedLSTMModel(nn.Module):
    """
    Better LSTM architecture bez sigmoid (regression!)
    
    Input: Mel-spectrograms (batch, n_mels, n_frames)
    Output: Unbounded [valence, arousal] - normalizujemy w post-processing
    """
    
    def __init__(self, n_mels: int = 128, hidden_size: int = 256, 
                 num_layers: int = 3, dropout: float = 0.4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention
        self.attention_query = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_key = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_value = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_scale = hidden_size ** 0.5
        
        # FC layers
        self.fc1 = nn.Linear(hidden_size, 512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout)
        
        # Output layer - NO SIGMOID (regression!)
        # Raw output [-inf, +inf]
        # Normalizujemy w post-processing z sigmoid
        self.output = nn.Linear(128, 2)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_mels, n_frames)
        Returns:
            (batch, 2) - raw [valence, arousal] values
        """
        # Transpose dla LSTM
        x = x.transpose(1, 2)  # (batch, n_frames, n_mels)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, n_frames, hidden_size*2)
        
        # Scaled dot-product attention
        Q = self.attention_query(lstm_out)
        K = self.attention_key(lstm_out)
        V = self.attention_value(lstm_out)
        
        scores = torch.matmul(Q, K.transpose(1, 2)) / self.attention_scale
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Pool
        context = context.mean(dim=1)
        
        # FC layers
        x = self.fc1(context)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Output - NO ACTIVATION (raw regression output)
        x = self.output(x)
        
        return x