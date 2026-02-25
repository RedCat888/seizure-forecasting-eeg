"""
Deep neural network for seizure forecasting.

Architecture:
1. Spectrogram CNN: Encodes log-magnitude STFT to embedding
2. Feature MLP: Encodes handcrafted features to embedding
3. Fusion Head: Combines embeddings for classification + regression
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..signal.spectrograms import SpectrogramTransform


class CNNEncoder(nn.Module):
    """
    CNN encoder for spectrogram input.
    
    Takes spectrogram [B, C, F, T] and outputs embedding [B, E].
    """
    
    def __init__(
        self,
        in_channels: int = 18,
        cnn_channels: List[int] = [32, 64, 128, 256],
        kernel_size: int = 3,
        pool_size: int = 2,
        embed_dim: int = 128,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_channels: Number of input EEG channels
            cnn_channels: Number of channels in each conv block
            kernel_size: Convolution kernel size
            pool_size: Max pooling size
            embed_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        
        # Build conv blocks
        layers = []
        prev_channels = in_channels
        
        for out_channels in cnn_channels:
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size),
                nn.Dropout2d(dropout / 2),
            ])
            prev_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Final embedding
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_channels[-1] * 4 * 4, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spectrogram [B, C, F, T]
            
        Returns:
            Embedding [B, E]
        """
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x


class FeatureMLP(nn.Module):
    """
    MLP encoder for handcrafted features.
    
    Takes features [B, F] and outputs embedding [B, E].
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [64, 64],
        embed_dim: int = 64,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_features: Number of input features
            hidden_dims: Hidden layer dimensions
            embed_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embed_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features [B, F]
            
        Returns:
            Embedding [B, E]
        """
        return self.mlp(x)


class FusionHead(nn.Module):
    """
    Fusion head that combines CNN and feature embeddings.
    
    Outputs:
    - Classification logit (binary)
    - Soft risk regression (sigmoid output)
    """
    
    def __init__(
        self,
        cnn_embed_dim: int,
        feature_embed_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
        use_features: bool = True,
    ):
        """
        Args:
            cnn_embed_dim: CNN embedding dimension
            feature_embed_dim: Feature MLP embedding dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            use_features: Whether to include feature embeddings
        """
        super().__init__()
        
        self.use_features = use_features
        
        if use_features:
            in_dim = cnn_embed_dim + feature_embed_dim
        else:
            in_dim = cnn_embed_dim
        
        # Shared MLP
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.shared_mlp = nn.Sequential(*layers)
        
        # Classification head
        self.cls_head = nn.Linear(prev_dim, 1)
        
        # Soft risk regression head
        self.soft_head = nn.Linear(prev_dim, 1)
    
    def forward(
        self,
        cnn_embed: torch.Tensor,
        feature_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cnn_embed: CNN embedding [B, E1]
            feature_embed: Feature embedding [B, E2] (optional)
            
        Returns:
            Tuple of (cls_logit [B, 1], soft_pred [B, 1])
        """
        if self.use_features and feature_embed is not None:
            x = torch.cat([cnn_embed, feature_embed], dim=1)
        else:
            x = cnn_embed
        
        x = self.shared_mlp(x)
        
        cls_logit = self.cls_head(x)
        soft_pred = torch.sigmoid(self.soft_head(x))
        
        return cls_logit, soft_pred


class FusionNet(nn.Module):
    """
    Complete fusion network for seizure forecasting.
    
    Combines:
    - SpectrogramTransform (raw EEG → spectrogram)
    - CNNEncoder (spectrogram → embedding)
    - FeatureMLP (handcrafted features → embedding)
    - FusionHead (embeddings → predictions)
    """
    
    def __init__(
        self,
        n_channels: int = 18,
        n_features: int = 40,
        cfg: Optional[DictConfig] = None,
        # Spectrogram params
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        # CNN params
        cnn_channels: List[int] = [32, 64, 128, 256],
        cnn_embed_dim: int = 128,
        # Feature MLP params
        feature_hidden_dims: List[int] = [64, 64],
        feature_embed_dim: int = 64,
        # Fusion params
        fusion_hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
        use_features: bool = True,
    ):
        """
        Args:
            n_channels: Number of EEG channels
            n_features: Number of handcrafted features
            cfg: Optional config (overrides other params)
            n_fft: STFT FFT size
            hop_length: STFT hop length
            win_length: STFT window length
            cnn_channels: CNN channel sizes
            cnn_embed_dim: CNN embedding dimension
            feature_hidden_dims: Feature MLP hidden layers
            feature_embed_dim: Feature MLP embedding dimension
            fusion_hidden_dims: Fusion head hidden layers
            dropout: Dropout rate
            use_features: Whether to use handcrafted features
        """
        super().__init__()
        
        # Override with config if provided
        if cfg is not None:
            spec_cfg = cfg.spectrogram
            model_cfg = cfg.model
            
            n_fft = spec_cfg.get("n_fft", n_fft)
            hop_length = spec_cfg.get("hop_length", hop_length)
            win_length = spec_cfg.get("win_length", win_length)
            
            cnn_channels = model_cfg.get("cnn_channels", cnn_channels)
            cnn_embed_dim = model_cfg.get("cnn_embed_dim", cnn_embed_dim)
            feature_hidden_dims = model_cfg.get("feature_hidden_dims", feature_hidden_dims)
            feature_embed_dim = model_cfg.get("feature_embed_dim", feature_embed_dim)
            fusion_hidden_dims = model_cfg.get("fusion_hidden_dims", fusion_hidden_dims)
            dropout = model_cfg.get("dropout", dropout)
            use_features = model_cfg.get("use_features", use_features)
        
        self.use_features = use_features
        
        # Spectrogram transform
        self.spectrogram = SpectrogramTransform(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        
        # CNN encoder
        self.cnn_encoder = CNNEncoder(
            in_channels=n_channels,
            cnn_channels=cnn_channels,
            embed_dim=cnn_embed_dim,
            dropout=dropout,
        )
        
        # Feature MLP
        if use_features:
            self.feature_mlp = FeatureMLP(
                in_features=n_features,
                hidden_dims=feature_hidden_dims,
                embed_dim=feature_embed_dim,
                dropout=dropout,
            )
        else:
            self.feature_mlp = None
        
        # Fusion head
        self.fusion_head = FusionHead(
            cnn_embed_dim=cnn_embed_dim,
            feature_embed_dim=feature_embed_dim if use_features else 0,
            hidden_dims=fusion_hidden_dims,
            dropout=dropout,
            use_features=use_features,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Raw EEG data [B, C, T]
            features: Handcrafted features [B, F] (optional)
            
        Returns:
            Tuple of (cls_logit [B, 1], soft_pred [B, 1])
        """
        # Compute spectrogram
        spec = self.spectrogram(x)  # [B, C, F, T']
        
        # CNN encoding
        cnn_embed = self.cnn_encoder(spec)
        
        # Feature encoding
        feature_embed = None
        if self.use_features and features is not None:
            feature_embed = self.feature_mlp(features)
        
        # Fusion
        cls_logit, soft_pred = self.fusion_head(cnn_embed, feature_embed)
        
        return cls_logit, soft_pred
    
    def predict_proba(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Raw EEG data [B, C, T]
            features: Handcrafted features [B, F] (optional)
            
        Returns:
            Probabilities [B]
        """
        cls_logit, _ = self.forward(x, features)
        return torch.sigmoid(cls_logit).squeeze(-1)


def create_model(
    n_channels: int,
    n_features: int,
    cfg: DictConfig,
) -> FusionNet:
    """
    Create FusionNet model from config.
    
    Args:
        n_channels: Number of EEG channels
        n_features: Number of handcrafted features
        cfg: Configuration
        
    Returns:
        FusionNet model
    """
    return FusionNet(
        n_channels=n_channels,
        n_features=n_features,
        cfg=cfg,
    )
