import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=6, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=6, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)
    

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs: torch.Tensor):
        # inputs shape (batch_size, channels, sequence_length)
        flat_input = inputs.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape[0], inputs.shape[2], self.embedding_dim)
        quantized = quantized.permute(0, 2, 1).contiguous()

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, vq_loss
    
    def get_indices(self, inputs: torch.Tensor):
        # inputs shape (batch, channels, time)
        flat_input = inputs.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1)
        # Retourne les indices sous la forme (batch, time)
        return encoding_indices.view(inputs.shape[0], inputs.shape[2])

    def quantize_from_indices(self, indices: torch.Tensor):
        # indices shape : (batch, time)
        quantized = self.embedding(indices) # (batch, time, embedding_dim)
        return quantized.permute(0, 2, 1).contiguous() # (batch, embedding_dim, time)


class VQAutoEncoder(nn.Module):
    def __init__(self, num_embeddings: int=256, embedding_dim: int=6):
        super().__init__()
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder()

    @staticmethod
    def _even_extend(x: torch.Tensor) -> torch.Tensor:
        """Concatenate signal with its time-reversal: [x, flip(x)]."""
        return torch.cat([x, torch.flip(x, dims=[-1])], dim=-1)

    def waveform_to_fft_real(self, x: torch.Tensor, *, pad_to_multiple: int = 8) -> Tuple[torch.Tensor, int]:
        """Waveform -> even-extension -> FFT -> real-valued spectrum.

        Returns (fft_real, pad_amount), where pad_amount is padding applied
        *after* even extension.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (batch, channels, time), got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected mono audio with 1 channel, got channels={x.shape[1]}")

        # If we need to pad for conv stride compatibility, pad *before* even-extension
        # so the resulting signal remains even (and the FFT stays (approximately) real).
        if pad_to_multiple % 2 != 0:
            raise ValueError(f"pad_to_multiple must be even (got {pad_to_multiple})")
        time_multiple = pad_to_multiple // 2

        original_len = x.shape[-1]
        pad_time = (-original_len) % time_multiple
        if pad_time:
            x = F.pad(x, (0, pad_time))

        x_even = self._even_extend(x)

        # FFT over time; even extension makes this (approximately) real-valued.
        fft_complex = torch.fft.fft(x_even.squeeze(1), dim=-1)
        fft_real = fft_complex.real.unsqueeze(1)
        return fft_real, pad_time

    def fft_real_to_waveform(
        self,
        fft_real: torch.Tensor,
        *,
        output_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Real spectrum -> iFFT -> waveform; crops to requested output length.

        If output_length is None, returns the first half of the even-extended
        signal (i.e. N/2 samples).
        """
        if fft_real.dim() != 3:
            raise ValueError(f"Expected fft_real shape (batch, channels, freq), got {tuple(fft_real.shape)}")
        if fft_real.shape[1] != 1:
            raise ValueError(f"Expected 1 channel for fft_real, got channels={fft_real.shape[1]}")

        fft_complex = torch.complex(fft_real.squeeze(1), torch.zeros_like(fft_real.squeeze(1)))
        time = torch.fft.ifft(fft_complex, dim=-1).real

        even_len = time.shape[-1]
        default_len = even_len // 2
        target_len = default_len if output_length is None else output_length
        return time[..., :target_len].unsqueeze(1)

    def forward(self, x: torch.Tensor, *, return_fft: bool = False):
        """Forward pass.

        Pipeline: waveform -> flip+concat -> FFT(real) -> VQ-AE -> iFFT -> crop.

        If return_fft=True, also returns reconstructed FFT(real) and target FFT(real).
        """
        original_length = x.shape[-1]
        fft_real, _pad_time = self.waveform_to_fft_real(x)

        latent = self.encoder(fft_real)
        quantized_latent, vq_loss = self.quantizer(latent)
        fft_reconstructed = self.decoder(quantized_latent)

        reconstructed = self.fft_real_to_waveform(
            fft_reconstructed,
            output_length=original_length,
        )

        if return_fft:
            return reconstructed, vq_loss, fft_reconstructed, fft_real
        return reconstructed, vq_loss

    def encode_to_indices(self, x: torch.Tensor):
        fft_real, _pad_time = self.waveform_to_fft_real(x)
        latent = self.encoder(fft_real)
        return self.quantizer.get_indices(latent)

    def decode_from_indices(self, indices: torch.Tensor, *, output_length: Optional[int] = None):
        quantized_latent = self.quantizer.quantize_from_indices(indices)
        fft_reconstructed = self.decoder(quantized_latent)
        return self.fft_real_to_waveform(fft_reconstructed, output_length=output_length)
