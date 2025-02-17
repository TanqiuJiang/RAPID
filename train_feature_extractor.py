import argparse, os, sys, datetime, glob
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
import argparse, os, sys, datetime
from time import perf_counter
from omegaconf import OmegaConf
from packaging import version
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from datasets import Dataset
from ldm.data.util import VirtualBatchWrapper
from ldm.util import instantiate_from_config
from ldm.privacy.myopacus import MyDPLightningDataModule
import torch
from tqdm import tqdm
from torchvision import transforms
from callbacks.cuda import CUDACallback                         # noqa: F401
from callbacks.image_logger import ImageLogger                  # noqa: F401
from callbacks.setup import SetupCallback                       # noqa: F401
from ldm.data.util import DataModuleFromConfig, WrappedDataset  # noqa: F401
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from ldm.models.diffusion.ddim import DDIMSampler
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset.__getitem__(i)
        label = torch.as_tensor(item["class_label"])
        item = torch.as_tensor(item["image"])
        item = item.permute(2, 0, 1)
        
        return item,label

def encoder_diffusion(sample_images,my_model):
    encoder_posterior = my_model.encode_first_stage(sample_images)
    z = my_model.get_first_stage_encoding(encoder_posterior).detach()

    return z

class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        
def train_model(config_path,ckptpath,model_output,model_epoch,model_batch_size):
    configs = [OmegaConf.load(config_path)][0]
    ckpt = ckptpath
    model = load_model_from_config(configs, ckpt)
    
    
    sampler = DDIMSampler(model)
    
    shape = (model.model.diffusion_model.in_channels,
                 model.model.diffusion_model.image_size,
                 model.model.diffusion_model.image_size)
    C,H,W = shape
    size = (model_batch_size,C,H,W)
    sampler.make_schedule(ddim_num_steps=100, ddim_eta=1, verbose=False)
    
    data_configs = [OmegaConf.load(config_path)][0]
    dataset_config = data_configs.data
    dataset = instantiate_from_config(dataset_config)
    dataset.prepare_data()
    dataset.setup()
    dataset = DatasetWrapper(dataset.datasets["validation"])
    trainset = dataset
    dataloader = DataLoader(dataset=dataset, batch_size=model_batch_size)
    
    net = Model(128).cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)
    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(32,scale=[0.6,1.0])])
    
    epochs = model_epoch
    for epoch in range(epochs):
        train_bar = tqdm(dataloader)
        total_loss, total_num = 0.0, 0
        for data,label in train_bar:
            net.train()
            with torch.no_grad():            
                view1_encoder = encoder_diffusion(train_transform(data).cuda(),model)
                view2_encoder = encoder_diffusion(train_transform(data).cuda(),model)
                my_noise = torch.randn_like(view1_encoder)
                #my_noise = torch.randn_like(view2_encoder)
                #my_noise2 = torch.randn_like(view2_encoder)
                step_number = int(torch.randint(800, 801, (1,))[0])
    
                ##forward
                timesteps = torch.randint(step_number, step_number + 1, (view1_encoder.shape[0],)).long().cuda()
                x_noisy_1 = model.q_sample(view1_encoder, timesteps, my_noise.cuda())
                #step_number = int(torch.randint(800, 990, (1,))[0])
                timesteps = torch.randint(step_number, step_number + 1, (view2_encoder.shape[0],)).long().cuda()
                my_noise2 = torch.randn_like(view2_encoder)
                x_noisy_2 = model.q_sample(view2_encoder, timesteps, my_noise2.cuda())
                
                condition = []
                for i in range (data.size(0)):
                    c = model.get_learned_conditioning({model.cond_stage_key: torch.tensor([label[i]]).cuda()})
                    condition.append(c)
                condition = torch.vstack(condition)
    
                recovered_samples, recovered_intermediates = sampler.ddim_sampling(condition, size,x_T=x_noisy_1.cuda(),
                                                        callback=None,
                                                        img_callback=None,
                                                        quantize_denoised=False,
                                                        mask=None, x0=None,
                                                        timesteps = None,
                                                        log_every_t=1,steps_needed=80,break_point=1
                                                        )
                #recovered1 = model.decode_first_stage(recovered_intermediates["pred_x0"][1])
                recovered1 = recovered_intermediates["pred_x0"][1]
                #print(recovered1.size())
                recovered_samples, recovered_intermediates = sampler.ddim_sampling(condition, size,x_T=x_noisy_2.cuda(),
                                                        callback=None,
                                                        img_callback=None,
                                                        quantize_denoised=False,
                                                        mask=None, x0=None,
                                                        timesteps = None,
                                                        log_every_t=1,steps_needed=80,break_point=1
                                                        )
                #recovered2 = model.decode_first_stage(recovered_intermediates["pred_x0"][1])
                recovered2 = recovered_intermediates["pred_x0"][1]
                
            # total_loss, total_num = 0.0, 0
            batch_size = data.size(0)
            temperature = 0.5
            feature_1, out_1 = net(recovered1)
            feature_2, out_2 = net(recovered2)
    
            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    
            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            # print(out)
            loss = (- torch.log(pos_sim / (sim_matrix.sum(dim=-1) + 1e-8))).mean()
            #print(loss.grad)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    torch.save(net,model_output)

def main(args):
    train_model(args.config, args.ckpt, args.output, args.epoch, args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--epoch", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    args = parser.parse_args()
    main(args)