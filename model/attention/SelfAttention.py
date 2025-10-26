import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================
# MODULE D'ATTENTION SELF-ATTENTION
# ========================================

class SelfAttention(nn.Module):
    """
    Module d'attention pour capturer les dépendances à longue distance
    Basé sur SAGAN (Self-Attention GAN)
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        
        # Réduire la dimension pour l'efficacité
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Paramètre appris pour balancer attention et features originales
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        x: input feature maps (B, C, H, W)
        """
        batch_size, C, H, W = x.size()
        
        # Query: (B, C//8, H*W)
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        
        # Key: (B, C//8, H*W)
        key = self.key_conv(x).view(batch_size, -1, H * W)
        
        # Attention map: (B, H*W, H*W)
        attention = self.softmax(torch.bmm(query, key))
        
        # Value: (B, C, H*W)
        value = self.value_conv(x).view(batch_size, -1, H * W)
        
        # Apply attention: (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection avec gamma appris
        out = self.gamma * out + x
        
        return out