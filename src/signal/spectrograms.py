"""
Spectrogram computation for deep learning models.

Uses PyTorch STFT for GPU acceleration during training.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramTransform(nn.Module):
    """
    Compute log-magnitude spectrogram from raw EEG using PyTorch STFT.
    
    Designed to be used as part of the model for GPU acceleration.
    """
    
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        log_offset: float = 1.0,
    ):
        """
        Args:
            n_fft: FFT size
            hop_length: Hop length between frames
            win_length: Window length
            log_offset: Offset for log transform (log(offset + mag))
        """
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.log_offset = log_offset
        
        # Register Hann window as buffer
        self.register_buffer(
            "window",
            torch.hann_window(win_length)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spectrogram for batch of multichannel signals.
        
        Args:
            x: Input tensor of shape [B, C, T] (batch, channels, time)
            
        Returns:
            Spectrogram tensor of shape [B, C, F, T'] where:
            - F = n_fft // 2 + 1 (frequency bins)
            - T' = (T - win_length) // hop_length + 1 (time frames)
        """
        batch_size, n_channels, n_times = x.shape
        
        # Reshape to process all channels together
        x_flat = x.reshape(batch_size * n_channels, n_times)
        
        # Compute STFT
        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=False,
        )
        
        # Get magnitude
        mag = torch.abs(stft)
        
        # Log transform
        log_mag = torch.log(self.log_offset + mag)
        
        # Reshape back to [B, C, F, T']
        n_freqs, n_frames = log_mag.shape[1], log_mag.shape[2]
        log_mag = log_mag.reshape(batch_size, n_channels, n_freqs, n_frames)
        
        return log_mag


def compute_spectrogram(
    signal: np.ndarray,
    sfreq: float,
    n_fft: int = 256,
    hop_length: int = 64,
    win_length: int = 256,
    log_offset: float = 1.0,
) -> np.ndarray:
    """
    Compute log-magnitude spectrogram using NumPy/SciPy.
    
    For offline preprocessing (caching).
    
    Args:
        signal: Input signal [C, T] or [T]
        sfreq: Sampling frequency
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        log_offset: Log offset
        
    Returns:
        Spectrogram array [C, F, T'] or [F, T']
    """
    from scipy import signal as sig
    
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False
    
    n_channels, n_times = signal.shape
    
    # Compute spectrogram for each channel
    specs = []
    for ch in range(n_channels):
        freqs, times, Sxx = sig.spectrogram(
            signal[ch],
            fs=sfreq,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            mode="magnitude",
        )
        
        # Log transform
        Sxx_log = np.log(log_offset + Sxx)
        specs.append(Sxx_log)
    
    result = np.stack(specs, axis=0)  # [C, F, T']
    
    if squeeze_output:
        result = result.squeeze(0)
    
    return result


def compute_spectrogram_batch(
    signals: np.ndarray,
    sfreq: float,
    n_fft: int = 256,
    hop_length: int = 64,
    win_length: int = 256,
    log_offset: float = 1.0,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute spectrograms for a batch of signals using GPU if available.
    
    Args:
        signals: Input signals [B, C, T]
        sfreq: Sampling frequency
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        log_offset: Log offset
        device: Device to use ('cuda', 'cpu', or None for auto)
        
    Returns:
        Spectrogram array [B, C, F, T']
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert to tensor
    x = torch.from_numpy(signals).float().to(device)
    
    # Create transform
    transform = SpectrogramTransform(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        log_offset=log_offset,
    ).to(device)
    
    # Compute
    with torch.no_grad():
        specs = transform(x)
    
    return specs.cpu().numpy()


def get_spectrogram_shape(
    n_times: int,
    n_channels: int,
    n_fft: int = 256,
    hop_length: int = 64,
    win_length: int = 256,
) -> Tuple[int, int, int]:
    """
    Get output shape of spectrogram without computing it.
    
    Args:
        n_times: Number of input time samples
        n_channels: Number of channels
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        
    Returns:
        Tuple of (n_channels, n_freqs, n_frames)
    """
    n_freqs = n_fft // 2 + 1
    n_frames = (n_times - win_length) // hop_length + 1
    
    return (n_channels, n_freqs, n_frames)
