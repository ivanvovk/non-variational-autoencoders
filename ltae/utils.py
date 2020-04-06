import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio


def gif_interpolation(model, dataloader, label_from, label_to, out_filename='new.gif'):
    '''
    Returns gif-file, which demonstrates the interpolation of trained autoencoder
    between two images by choosing random objects according to provided class labels. 
    '''
    # for quick detach of cuda Tensors
    cdn = lambda x: x.cpu().detach().numpy()
    device = list(model.parameters())[0].device

    # get batch (assumes the whole dataset is extracted)
    x, y = next(iter(dataloader))
    x = x.view(-1, 28 * 28).to(device)
    
    # get random images of provided classes
    IDX_FROM = np.random.choice(np.where(y == label_from)[0])
    IDX_TO = np.random.choice(np.where(y == label_to)[0])

    # interpolation steps
    alphas = torch.FloatTensor(np.arange(0.1, 1.1, 0.1))

    _, latent_code, _, _ = model(x)

    # reconstruct interpolated images
    reconstructions = []
    reconstructions.append(cdn(x[IDX_FROM]))
    for alpha in alphas:
        interpolated_code = latent_code[IDX_FROM] + alpha * (latent_code[IDX_TO] - latent_code[IDX_FROM])
        eps = model.generate_eps()
        eps = torch.FloatTensor(eps).to(device)
        reconstructions.append(cdn(model.decoder(interpolated_code + eps)))
    reconstructions.append(cdn(x[IDX_TO]))
    
    # create folder to save interpolated images
    DIR_FOR_PICS = 'pics/interpolation/'
    if not os.path.isdir(DIR_FOR_PICS):
        os.makedirs(DIR_FOR_PICS)
        os.chmod(DIR_FOR_PICS, 0o775)
    
    # save images
    for i in range(len(reconstructions)):
        plt.imshow(reconstructions[i].reshape(28, 28), cmap='gray') 
        plt.savefig(os.path.join(DIR_FOR_PICS, '{}.png'.format(str(i).zfill(2))))
    
    paths = [os.path.join(DIR_FOR_PICS, path) for path in sorted(os.listdir(DIR_FOR_PICS)) if '.ipynb_checkpoints' not in path]
    
    # create gif
    images = []
    for filename in paths:
        images.append(imageio.imread(filename))
    imageio.mimsave(out_filename, images)


def random_interpolation(autoencoder, loader, latent=True, savepath=None):
    cdn = lambda x: x.cpu().detach().numpy()
    
    device = list(autoencoder.parameters())[0].device
    
    batch, _ = next(iter(loader))
    batch = batch.view(-1, 28 * 28).to(device)
    bs = batch.shape[0]
    
    alphas = 0.5 * torch.rand(bs, 1).to(device)
    
    if latent:
        _, latent_code, _, reconstruction = autoencoder(batch)
    else:
        latent_code, reconstruction = autoencoder(batch)
    
    shifted_index = torch.arange(0, bs) - 1
    interpolated_code = latent_code + alphas * (latent_code[shifted_index] - latent_code)
        
    interpolation = cdn(autoencoder.decoder(interpolated_code))
    
    def img_reshape(img):
        if len(img.shape) == 1:
            return img.reshape(28, 28)
        elif len(img.shape) == 3:
            return img.reshape(-1, 28, 28)
        
    top, bottom = cdn(batch[:5]), cdn(batch[shifted_index][:5])
    top_reconstruction, bottom_reconstruction = cdn(reconstruction[:5]), cdn(reconstruction[shifted_index][:5]) 
        
    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(10, 10))
    
    for col in range(5):
        alpha = np.round(alphas[col].cpu().numpy().item(), 3)
        
        ax[0, col].set_title(f'Alpha: {alpha}', size=14)
        
        for v in range(5):
            ax[v, col].axis('off')

        ax[0, col].imshow(img_reshape(top[col]), cmap=plt.cm.gray)
        ax[1, col].imshow(img_reshape(top_reconstruction[col]), cmap=plt.cm.gray)
        
        ax[2, col].imshow(img_reshape(interpolation[col]), cmap=plt.cm.gray)
        
        ax[3, col].imshow(img_reshape(bottom_reconstruction[col]), cmap=plt.cm.gray)
        ax[4, col].imshow(img_reshape(bottom[col]), cmap=plt.cm.gray)
        
    plt.tight_layout(pad=0.22)
    
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()
    
    
def draw_reconstruction(model, dataset, num_img):
    fig, ax = plt.subplots(2, num_img, figsize=(10, 4))
    
    IDX = np.random.choice(len(dataset.data), num_img)
    view_data = dataset.data[IDX].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    for i in range(num_img):
        ax[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray') 
        ax[0][i].set_xticks(()); ax[0][i].set_yticks(())

    decoded_data = model(view_data.cuda())[-1]
    for i in range(num_img):
        ax[1][i].imshow(np.reshape(decoded_data.detach().cpu().numpy()[i], (28, 28)), cmap='gray')
        ax[1][i].set_xticks(()); ax[1][i].set_yticks(())