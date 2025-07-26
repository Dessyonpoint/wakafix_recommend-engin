import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoFillEmbeddingNN(nn.Module):
    def __init__(self, n_service, n_location, n_time, embedding_dim, hidden_size, output_size):
        super(AutoFillEmbeddingNN, self).__init__()
        self.service_embed = nn.Embedding(n_service, embedding_dim)
        self.location_embed = nn.Embedding(n_location, embedding_dim)
        self.time_embed = nn.Embedding(n_time, embedding_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, s, l, t):
        s_embed = self.service_embed(s)
        l_embed = self.location_embed(l)
        t_embed = self.time_embed(t)

        x = torch.cat([s_embed, l_embed, t_embed], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)
