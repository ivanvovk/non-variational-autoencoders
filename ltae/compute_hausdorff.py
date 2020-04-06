import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

from model import AutoEncoder, LatentAutoEncoder, VariationalAE
from utils import gif_interpolation, draw_reconstruction, random_interpolation


if __name__ == '__main__':
    train_data = torchvision.datasets.MNIST(
        root='datasets/mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    autoencoders = []
    PATH = 'saved_models/comparison/standard/'
    model = LatentAutoEncoder(2, 'standard', sigma=sigma).cuda().eval()
    model.set_device(); 
    checkpoint_dict = torch.load(os.path.join(PATH, model_path), map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
        
    metrics = []
    for _ in range(10):
        distances = []
        for step, (x, _) in enumerate(Data.DataLoader(dataset=train_data, batch_size=1000, shuffle=True)):
            x_batch = x.view(-1, 28 * 28)
            sample = torch.randn(1000, 2).cuda()
            sample = model.decode(sample).detach().cpu()
            distances.append(directed_hausdorff(x_batch.numpy(), sample.numpy()))
        metrics.append(np.array([dist[0] for dist in distances]).mean())
    metrics = np.array(metrics)
    return metrics.mean(), metrics.std()