import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        quantized_latent, vq_loss = self.quantizer(latent)
        reconstructed = self.decoder(quantized_latent)
        return reconstructed, vq_loss

    def encode_to_indices(self, x: torch.Tensor):
        latent = self.encoder(x)
        return self.quantizer.get_indices(latent)

    def decode_from_indices(self, indices: torch.Tensor):
        quantized_latent = self.quantizer.quantize_from_indices(indices)
        return self.decoder(quantized_latent)
