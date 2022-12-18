"""
VAE model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Implement the VAE model with BCE loss and KL
    """
    def __init__(self, latent_dim, device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )

    def sample(self, size):
        """
        :param size: size of the sample
        :return: reconstructed images
        """
        with torch.no_grad():
            x_sample = torch.rand((size, self.latent_dim)).to(self.device)
            x_reconstruct = self.decoder(self.upsample(x_sample).view(-1, 64, 7, 7)).to(self.device)
            return x_reconstruct

    def z_sample(self, mu, logvar):
        """
        :param mu: expectation of the distribution
        :param logvar: log variance of the distribution
        :return: vector of gaussian random variable with e=mu and sigma = std
        """
        std = torch.exp(logvar / 2)
        eps = torch.rand_like(std).to(self.device)
        return mu + eps * std

    @staticmethod
    def loss(x, recon, mu, logvar):
        """
        :param x: original image
        :param recon: reconstructed image
        :param mu: expectation of the distribution
        :param logvar: log variance of the distribution
        :return: VAE loss (BCE + KL)
        """
        # calculate the KL divergence
        KL = -1/2 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # calculate the binary cross entropy loss
        BCE = F.binary_cross_entropy(recon, x, reduction='sum')
        return BCE + KL

    def forward(self, x):
        """
        :param x: original image
        :return: forward pass of the VAE model
        """
        x_latent = self.encoder(x).view(-1, 64*7*7)
        mu = self.mu(x_latent)
        logvar = self.logvar(x_latent)
        z = self.z_sample(mu=mu, logvar=logvar)
        return self.decoder(self.upsample(z).view(-1, 64, 7, 7)), mu, logvar
