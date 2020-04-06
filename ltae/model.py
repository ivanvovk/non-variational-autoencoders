import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.nn.parameter import Parameter


class AutoEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.Softplus(),
            nn.Linear(500, 500),
            nn.Softplus(),
            nn.Linear(500, hidden_size),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 500),
            nn.Softplus(),
            nn.Linear(500, 500),
            nn.Softplus(),
            nn.Linear(500, 28 * 28),
            nn.Sigmoid(), 
        )
    
    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded


class LatentAutoEncoder(nn.Module):
    def __init__(self, hidden_size, norm_func='standard', sigma=0.01):
        super(LatentAutoEncoder, self).__init__()
        
        self.alpha = Parameter(torch.empty((1, hidden_size)))
        self.beta = Parameter(torch.empty((1, hidden_size)))
        torch.nn.init.xavier_normal_(self.alpha)
        torch.nn.init.xavier_normal_(self.beta)

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.Softplus(),
            nn.Linear(500, 500),
            nn.Softplus(),
            nn.Linear(500, hidden_size),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 500),
            nn.Softplus(),
            nn.Linear(500, 500),
            nn.Softplus(),
            nn.Linear(500, 28 * 28),
            nn.Sigmoid(), 
        )
        
        if norm_func not in ['standard', 'min-max']:
            raise ValueError('Wrong type of normalization. Only allowed: [\"standard\", \"min-max\"].')
        elif norm_func == 'standard':
            self.norm_func = lambda z: (z - z.mean(dim=1, keepdim=True)) / z.std(dim=1, keepdim=True)
            self.generate_eps = lambda: np.random.normal(scale=sigma, size=hidden_size)
        elif norm_func == 'min-max':
            self.norm_func = lambda z: (z - z.min()) / (z.max() - z.min())
            self.generate_eps = lambda: np.random.uniform(-sigma, sigma, size=hidden_size)
    
    def set_device(self):
        self.device = list(self.parameters())[0].device
    
    def get_device(self):
        return self.device

    def transform(self, z):
        return self.norm_func(z)
        
    def latent(self, z):
        return self.alpha * (z + self.beta)
    
    def forward(self, x):
        x_encoded = self.encoder(x)
        
        x_latent = self.latent(x_encoded)
        x_transformed = self.transform(x_encoded)
        
        # generate set of noise
        eps = np.vstack([self.generate_eps() for _ in range(x.shape[0])])
        eps = torch.FloatTensor(eps).to(self.device)
        
        x_decoded = self.decoder(x_latent + eps)
        return x_encoded, x_latent, x_transformed, x_decoded
    
    
class VariationalAE(nn.Module):
    def __init__(self, hidden_size):
        super(VariationalAE, self).__init__()

        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, hidden_size)
        self.fc22 = nn.Linear(500, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 500)
        self.fc4 = nn.Linear(500, 784)

    def encode(self, x):
        h1 = F.softplus(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.softplus(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
