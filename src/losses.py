import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleModalCox(nn.Module):
    """
    MLP for Single-Modal Cox-MT (RNA or WSI).
    Structure: Input -> Linear -> ReLU -> Dropout -> ... -> Linear -> Output
    """
    def __init__(self, input_dim, hidden_dims=[1000, 200], dropout_rate=0.2):
        super(SingleModalCox, self).__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        # Final output layer (Hazard Ratio)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MultiModalFusion(nn.Module):
    """
    Implements the Cross-Attention Tokenization described in Figure 5b.
    """
    def __init__(self, rna_dim=4000, wsi_dim=1024, embed_dim=256, num_heads=4):
        super(MultiModalFusion, self).__init__()
        
        # 1. Tokenization Projections [cite: 196-198]
        self.rna_proj = nn.Linear(rna_dim, embed_dim)
        self.wsi_proj = nn.Linear(wsi_dim, embed_dim)
        
        # 2. Multi-Head Attention Units [cite: 195]
        # Mutual Attention: Queries are swapped
        self.mha1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 3. Final Prediction MLP
        # Input is concatenated output of two MHAs (256 + 256 = 512)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 1)
        )

    def forward(self, x_rna, x_wsi_seq):
        """
        x_rna: (Batch, 4000)
        x_wsi_seq: (Batch, Seq_Len, 1024) - Sequence of patch features
        """
        # Tokenize
        # RNA: (B, 4000) -> (B, 1, 256)
        t_rna = self.rna_proj(x_rna).unsqueeze(1)
        
        # WSI: (B, Seq, 1024) -> (B, Seq, 256)
        t_wsi = self.wsi_proj(x_wsi_seq)
        
        # Mutual Attention [cite: 201-202]
        # MHA1: Query=WSI, Key/Val=RNA (Broadcasting RNA to WSI sequence length)
        # Note: Paper implies generating query from one seq and attending to other
        # Assuming we want a global representation, we use the CLS token or average.
        # Below implements standard cross attention:
        
        # Query: RNA, Key/Value: WSI
        attn_out1, _ = self.mha1(query=t_rna, key=t_wsi, value=t_wsi)
        
        # Query: WSI (pooled/CLS), Key/Value: RNA
        # We pool WSI to single token for query
        t_wsi_pooled = torch.mean(t_wsi, dim=1, keepdim=True)
        attn_out2, _ = self.mha2(query=t_wsi_pooled, key=t_rna, value=t_rna)
        
        # Concatenate outputs
        # (B, 1, 256) + (B, 1, 256) -> (B, 512)
        fusion_feat = torch.cat([attn_out1.squeeze(1), attn_out2.squeeze(1)], dim=1)
        
        return self.predictor(fusion_feat)
