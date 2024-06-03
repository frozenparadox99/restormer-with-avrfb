import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import glob
import os
import random
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import torchvision

import os

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


train_dir = './Restormer/Training_Dataset'
test_dir = './Restormer/Proper_Test'

output_dir = './output'

args = {
    'data_path': train_dir,
    'data_path_test': test_dir,
    'data_name': 'rain100H',
    'save_path': output_dir,
    'num_blocks': [4, 6, 6, 8],
    'num_heads': [1, 2, 4, 8],
    'channels': [48, 96, 192, 384],
    'expansion_factor': 2.66,
    'num_refinement': 4,
    'num_iter':9057800,
    'batch_size': [2, 2, 2, 2, 2, 2],
    'patch_size': [128, 160, 192, 256, 320, 384],
    'lr': 0.0003,
    'milestone': [276000],
    'workers': 0,
    'seed': -1,
    'model_file': None,
    'finetune': torch._functional_sym_constrain_range_for_size
}


class AdaptiveVaryingReceptiveFusionBlock(nn.Module):
    def __init__(self, in_channels, init_value=-0.80):
        super(AdaptiveVaryingReceptiveFusionBlock, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=3, dilation=3, bias=False)
        self.conv5 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=5, dilation=5, bias=False)
        self.conv_final = nn.Conv2d(16, in_channels, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.weight = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def adaptive_mixup_feature_fusion_block(self, feature1, feature2):
        sig = torch.sigmoid(self.weight)
        mix_factor = sig
        out = feature1 * mix_factor + feature2 * (1 - mix_factor)
        return out

    def forward(self, x):
        conv3_out = self.relu(self.conv3(x))
        conv5_out = self.relu(self.conv5(x))
        concatenated = self.adaptive_mixup_feature_fusion_block(conv3_out, conv5_out)
        out = self.relu(self.conv_final(concatenated))
        return out

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # Adaptive Varying Receptive Fusion Block
        self.avrf_block = AdaptiveVaryingReceptiveFusionBlock(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))

        # Apply the AVRF block
        out = self.avrf_block(out)

        return out

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Restormer(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384], num_refinement=4,
                 expansion_factor=2.66):
        super(Restormer, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out

class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_path_test = args.data_path_test
        self.data_name = args.data_name
        self.save_path = args.save_path
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.channels = args.channels
        self.expansion_factor = args.expansion_factor
        self.num_refinement = args.num_refinement
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.milestone = args.milestone
        self.workers = args.workers
        self.model_file = args.model_file
        self.finetune = args.finetune


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return Config(args)


def pad_image_needed(img, size):
    width, height = T.get_image_size(img)
    if width < size[1]:
        img = T.pad(img, [size[1] - width, 0], padding_mode='reflect')
    if height < size[0]:
        img = T.pad(img, [0, size[0] - height], padding_mode='reflect')
    return img


class RainDataset(Dataset):
    def __init__(self, data_path, data_path_test, data_name, data_type, patch_size=None, length=None):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = data_name, data_type, patch_size
        self.rain_images = sorted(glob.glob('{}/*.jpg'.format(data_path)))
        self.rain_images_test = sorted(glob.glob('{}/*.jpg'.format(data_path_test)))
        # self.rain_images = sorted(glob.glob('{}/{}/{}/inp/*.png'.format(data_path, data_name, data_type)))
        # self.norain_images = sorted(glob.glob('{}/{}/{}/tar/*.png'.format(data_path, data_name, data_type)))
        # make sure the length of training and testing different
        self.num = len(self.rain_images)

        self.num_test = len(self.rain_images_test)
        self.sample_num = length if data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):


        if self.data_type == 'train':
            image_name = os.path.basename(self.rain_images[idx % self.num])

            imag = np.array(Image.open(self.rain_images[idx % self.num]))
            r,c,ch = imag.shape
            # width = c//3
            width = c//2

            rain = T.to_tensor(imag[:,:width,:])
            norain = T.to_tensor(imag[:,width:width*2,:])
            # mask = T.to_tensor(imag[:,width*2:width*3,:])

            # print('rain shape..............',rain.shape)
            # print('mask shape..............',mask.shape)
            # exit(0)
            # norain = T.to_tensor(Image.open(self.norain_images[idx % self.num]))
            h, w = rain.shape[1:]

            # make sure the image could be cropped
            rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            # mask = pad_image_needed(mask, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
            # mask = T.crop(mask, i, j, th, tw)
            if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
                # mask = T.hflip(mask)
            if torch.rand(1) < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)
                # mask = T.vflip(mask)


        else:
            image_name = os.path.basename(self.rain_images_test[idx % self.num_test])

            imag = np.array(Image.open(self.rain_images_test[idx % self.num_test]))

            r,c,ch = imag.shape
            # width = c//3
            width = c//2

            rain = T.to_tensor(imag[:,:width,:])
            norain = T.to_tensor(imag[:,width:width*2,:])
            # mask = T.to_tensor(imag[:,width*2:width*3,:])

            # print(rain.shape)
            # exit(0)
            # norain = T.to_tensor(Image.open(self.norain_images[idx % self.num]))
            h, w = rain.shape[1:]
            # padding in case images are not multiples of 8
            # new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
            # pad_h = new_h - h if h % 8 != 0 else 0
            # pad_w = new_w - w if w % 8 != 0 else 0
            # rain = F.pad(rain, (0, pad_w, 0, pad_h), 'reflect')
            # mask = F.pad(mask, (0, pad_w, 0, pad_h), 'reflect')
            # norain = F.pad(norain, (0, pad_w, 0, pad_h), 'reflect')
        # return rain, norain, mask, image_name, h, w
        return rain, norain, image_name, h, w


def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def psnr(x, y, data_range=255.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = - 10 * torch.log10(mse)
    return score


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    
perceptual_loss = VGGPerceptualLoss().cuda()
results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0

def test_loop(net, data_loader, num_iter):
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        # for rain, norain, mask, name, h, w in test_bar:
        for rain, norain, name, h, w in test_bar:
            # rain, norain, mask = rain.cuda(), norain.cuda(), mask.cuda()
            rain, norain = rain.cuda(), norain.cuda()
            # out = torch.clamp((torch.clamp(model(rain, mask)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            out = torch.clamp((torch.clamp(model(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255).byte()
            # computer the metrics with Y channel and double precision
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1
            save_path = '{}/{}/{}'.format(args["save_path"], args["data_name"], name[0])
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(save_path)
            test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                     .format(num_iter, 1 if args["model_file"] else args["num_iter"],
                                             total_psnr / count, total_ssim / count))
    return total_psnr / count, total_ssim / count


def save_loop(net, data_loader, num_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append('{:.2f}'.format(val_psnr))
    results['SSIM'].append('{:.3f}'.format(val_ssim))
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, (num_iter if args["model_file"] else num_iter // 1000) + 1))
    data_frame.to_csv('{}/{}.csv'.format(args["save_path"], args["data_name"]), index_label='Iter', float_format='%.3f')
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open('{}/{}.txt'.format(args["save_path"], args["data_name"]), 'w') as f:
            f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr, best_ssim))
        torch.save(model.state_dict(), '{}/{}.pth'.format(args["save_path"], args["data_name"]))


test_dataset = RainDataset(args["data_path_test"], args["data_path_test"], args["data_name"], 'test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args["workers"])
# train_dataset = RainDataset(args["data_path"], args["data_path_test"], args["data_name"], 'train', args["patch_size"][i], length)
# train_loader = DataLoader(train_dataset, args["batch_size"][i], True, num_workers=args["workers"])

model = Restormer(args["num_blocks"], args["num_heads"], args["channels"], args["num_refinement"], args["expansion_factor"]).cuda()
if args["model_file"]:
    model.load_state_dict(torch.load(args["model_file"]))
    save_loop(model, test_loader, 1)
else:
    optimizer = AdamW(model.parameters(), lr=args["lr"], weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args["num_iter"], eta_min=1e-6)
    total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
    train_bar = tqdm(range(1, args["num_iter"] + 1), initial=1, dynamic_ncols=True)
    for n_iter in train_bar:
        # progressive learning
        if n_iter == 1 or n_iter - 1 in args["milestone"]:
            end_iter = args["milestone"][i] if i < len(args["milestone"]) else args["num_iter"]
            start_iter = args["milestone"][i - 1] if i > 0 else 0
            length = args["batch_size"][i] * (end_iter - start_iter)
            train_dataset = RainDataset(args["data_path"], args["data_path_test"], args["data_name"], 'train', args["patch_size"][i], length)
            train_loader = iter(DataLoader(train_dataset, args["batch_size"][i], True, num_workers=args["workers"]))
            i += 1
        # train
        model.train()
        # rain, norain, mask, name, h, w = next(train_loader)
        # rain, norain, mask = rain.cuda(), norain.cuda(), mask.cuda()
        # out = model(rain, mask)
        rain, norain, name, h, w = next(train_loader)
        rain, norain = rain.cuda(), norain.cuda()
        out = model(rain)
        loss = F.l1_loss(out, norain)+ perceptual_loss(out, norain) * 0.2


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_num += rain.size(0)
        total_loss += loss.item() * rain.size(0)
        train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                  .format(n_iter, args["num_iter"], total_loss / total_num))

        lr_scheduler.step()
        if n_iter % 1000 == 0:
            results['Loss'].append('{:.3f}'.format(total_loss / total_num))
            save_loop(model, test_loader, n_iter)
