import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim=132, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256,
                 num_classes=6, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        cls_token = torch.zeros(B, 1, self.d_model, device=x.device)
        x = self.embedding(x)                     # [B, T, d_model]
        x = torch.cat([cls_token, x], dim=1)      # [B, T+1, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]                            # CLS token 输出
        x = self.classifier(x)
        return x