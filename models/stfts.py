import torch
import torch.nn as nn

def mag_phase_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=False, addeps=False):
    """
    Compute magnitude and phase using STFT.

    Args:
        y (torch.Tensor): Input audio signal.
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        center (bool, optional): Whether to center the signal before padding. Defaults to False (causal).
        eps (bool, optional): Whether adding epsilon to magnitude and phase or not. Defaults to False. 

    Returns:
        tuple: Magnitude, phase, and complex representation of the STFT.
    """
    #eps = torch.finfo(y.dtype).eps
    eps = 1e-10
    hann_window = torch.hann_window(win_size).to(y.device)
    if center is False:
        hann_window = hann_window + 1e-3
    stft_spec = torch.stft(
                    y, n_fft, 
                    hop_length=hop_size, 
                    win_length=win_size, 
                    window=hann_window,
                    center=center, 
                    pad_mode='reflect', 
                    normalized=False, 
                    return_complex=True)

    if addeps==False:
        mag = torch.abs(stft_spec)
        pha = torch.angle(stft_spec)
    else:
        real_part = stft_spec.real
        imag_part = stft_spec.imag
        mag = torch.sqrt(real_part.pow(2) + imag_part.pow(2) + eps)
        pha = torch.atan2(imag_part + eps, real_part + eps)
    # Compress the magnitude (clamp to avoid inf gradients when compress_factor < 1)
    mag = torch.pow(torch.clamp(mag, min=eps), compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_phase_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=False, length=None):
    """
    Inverse STFT to reconstruct the audio signal from magnitude and phase.

    Args:
        mag (torch.Tensor): Magnitude of the STFT.
        pha (torch.Tensor): Phase of the STFT.
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        center (bool, optional): Whether to center the signal before padding. Defaults to False (causal).

    Returns:
        torch.Tensor: Reconstructed audio signal.
    """
    eps = 1e-10
    mag = torch.pow(torch.clamp(mag, min=eps), 1.0 / compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    # Avoid zero window_sum when center=False by adding small epsilon at the edges
    if not center:
        hann_window = hann_window + 1e-3
    target_length = length if length is not None else (mag.size(-1) * hop_size + win_size - hop_size)
    wav = torch.istft(
                    com, 
                    n_fft, 
                    hop_length=hop_size, 
                    win_length=win_size, 
                    window=hann_window, 
                    center=center,
                    length=target_length)
    return wav
