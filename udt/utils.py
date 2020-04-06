import torch.nn as nn
from torchvision import transforms
import torch
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import Dataset

import pandas as pd


def get_test_imgs(args):
    comp_transform = transforms.Compose([
        transforms.CenterCrop(args.crop),
        transforms.Resize(args.resize),
        transforms.ToTensor()
    ])

    domA_test = CustomDataset(args.data_path + 'testA.txt', transform=comp_transform)
    domB_test = CustomDataset(args.data_path + 'testB.txt', transform=comp_transform)

    domA_test_loader = torch.utils.data.DataLoader(domA_test, batch_size=64,
                                                   shuffle=False, num_workers=6)
    domB_test_loader = torch.utils.data.DataLoader(domB_test, batch_size=64,
                                                   shuffle=False, num_workers=6)

    for domA_img in domA_test_loader:
        domA_img = domA_img.to(args.device)            
        break

    for domB_img in domB_test_loader:
        domB_img = domB_img.to(args.device)
        break

    return domA_img, domB_img


def save_imgs(args, e1, e2, decoder, iters):
    test_domA, test_domB = get_test_imgs(args)
    test_domA, test_domB = test_domA.to(args.device), test_domB.to(args.device)
    
    exps = []

    for i in range(args.num_display):
        with torch.no_grad():
            if i == 0:
                filler = test_domB[i].unsqueeze(0).clone()
                exps.append(filler.fill_(0))

            exps.append(test_domB[i].unsqueeze(0))

    for i in range(args.num_display):
        exps.append(test_domA[i].unsqueeze(0))
        separate_A = e2(test_domA[i].unsqueeze(0))
        for j in range(args.num_display):
            with torch.no_grad():
                common_B = e1(test_domB[j].unsqueeze(0))

                BA_encoding = torch.cat([common_B, separate_A], dim=1)
                BA_decoding = decoder(BA_encoding)
                exps.append(BA_decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/experiments_%06d.png' % (args.out, iters),
                      normalize=True, nrow=args.num_display + 1)


def save_model(out_file, e1, e2, decoder, ae_opt, disc, disc_opt, iters):
    state = {
        'e1': e1.state_dict(),
        'e2': e2.state_dict(),
        'decoder': decoder.state_dict(),
        'ae_opt': ae_opt.state_dict(),
        'disc': disc.state_dict(),
        'disc_opt': disc_opt.state_dict(),
        'iters': iters
    }
    torch.save(state, out_file)
    return


def load_model(load_path, e1, e2, decoder, ae_opt, disc, disc_opt):
    state = torch.load(load_path)
    e1.load_state_dict(state['e1'])
    e2.load_state_dict(state['e2'])
    decoder.load_state_dict(state['decoder'])
    ae_opt.load_state_dict(state['ae_opt'])
    disc.load_state_dict(state['disc'])
    disc_opt.load_state_dict(state['disc_opt'])
    return state['iters']


def default_loader(path):
    return Image.open(path).convert('RGB')


    
class CustomDataset(Dataset):
    def __init__(self, path, transform=None, return_paths=False,
                 loader=default_loader):
        super(CustomDataset, self).__init__()

        with open(path) as f:
            imgs = [s.replace('\n', '') for s in f.readlines()]

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + path + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img).mul(2).add(-1)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)