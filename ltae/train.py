import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import functional as F
import torchvision

from model import AutoEncoder, LatentAutoEncoder, VariationalAE


EPOCH = 1
BATCH_SIZE = 100


def loss_func_vae(recon_x, x, mu, logvar, l2=False):
    if l2:
        left_term = F.mse_loss(recon_x, x.view(-1, 784))
    else:
        left_term = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return left_term + KLD


def train(output_filename, model_type, hidden_size, loss_type, norm_type, sigma_noise):
    train_data = torchvision.datasets.MNIST(
        root='datasets/mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    if loss_type == 'l2':
        loss_func = nn.MSELoss()
    elif loss_type == 'cross_entropy':
        loss_func = F.binary_cross_entropy
        
    if model_type == 'AE':
        model = AutoEncoder(hidden_size).cuda()
    elif model_type == 'LTAE':
        model = LatentAutoEncoder(hidden_size, norm_type, sigma=sigma_noise).cuda()
        model.set_device()
    elif model_type == 'VAE':
        model = VariationalAE(hidden_size).cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(EPOCH):
        for step, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()

            x_batch = x.view(-1, 28 * 28).cuda()
            y_batch = x.view(-1, 28 * 28).cuda()

            if model_type == 'AE':
                _, decoded = model(x_batch)
                loss = loss_func(decoded, y_batch) 
            elif model_type == 'LTAE':
                _, latent, transformed, decoded = model(x_batch)
                loss = loss_func(decoded, y_batch) 
                loss += torch.nn.functional.mse_loss(transformed, latent)
            elif model_type == 'VAE':
                decoded, mu, logvar = model(x_batch)
                loss = loss_func_vae(decoded, x_batch, mu, logvar, loss_type)

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0: print('Epoch: ', epoch, '| train loss: %.4f' % loss.detach().cpu())

    torch.save({'state_dict': model.state_dict()}, f'./saved_models/{output_filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_filename', type=str,
                        help='filename to save checkpoint')
    parser.add_argument('-m', '--model_type', type=str, help='model type from AE, LTAE, VAE')
    parser.add_argument('--hidden_size', type=int, help='bottleneck size')
    parser.add_argument('-l', '--loss_type', type=str, help='l2 or cross entropy')
    parser.add_argument('-n', '--norm_type', type=str,
                        default=None, required=False, help='normalization technique')
    parser.add_argument('-s', '--sigma_noise', type=float,
                        default=None, required=False, help='sigma for the noise')
    

    args = parser.parse_args()

    train(args.output_filename, args.model_type, args.hidden_size, args.loss_type, args.norm_type, args.sigma_noise)

    