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


def unlabeled_sampling(model_batch_size,repeat,model,sampler,private_sampler,net,feature_bank,train_data):
    shape = (model.model.diffusion_model.in_channels,
                 model.model.diffusion_model.image_size,
                 model.model.diffusion_model.image_size)
    C,H,W = shape
    for l in range (1):
        for i in range (repeat):
            size = (model_batch_size,C,H,W)
            condition = []
            for k in range (model_batch_size):
                condition.append(None)
            condition = torch.vstack(condition)
            my_samples, my_intermediates = sampler.ddim_sampling(condition, size,
                                                            callback=None,
                                                            img_callback=None,
                                                            quantize_denoised=False,
                                                            mask=None, x0=None,
                                                            timesteps = None,
                                                            log_every_t=1
                                                            )
            noisy_key = my_intermediates["pred_x0"][20]
            feature, out = net(noisy_key.cuda())
            sim_weight, sim_indices = torch.mm(feature, feature_bank.t().contiguous()).topk(k=1, dim=-1)
            for j in range (len(sim_indices)):
                if j == 0: 
                    noisy_image = train_data[sim_indices[j][0]][0].unsqueeze(0)
                else:
                    noisy_image = torch.cat((noisy_image,train_data[sim_indices[j][0]][0].unsqueeze(0)),0)
            timestep = torch.randint(300, 301, (len(noisy_image),)).long()
            encoder_posterior = model.encode_first_stage(noisy_image.cuda())
            new_latent = model.get_first_stage_encoding(encoder_posterior).detach()
            single_noise = torch.randn_like(new_latent)
            new_noisy_latent = model.q_sample(new_latent.cuda(),timestep.cuda(),single_noise.cuda()).detach().cpu()
            recovered_samples, recovered_intermediates = private_sampler.ddim_sampling(condition, size,x_T=new_noisy_latent.cuda(),
                                                            callback=None,
                                                            img_callback=None,
                                                            quantize_denoised=False,
                                                            mask=None, x0=None,
                                                            timesteps = None,
                                                            log_every_t=1,steps_needed=30
                                                            )
            my_image = torch.clip(model.decode_first_stage(recovered_intermediates["pred_x0"][30]),-1,1)
            if i == 0:
                final_latent = my_image
            else:
                final_latent = torch.cat((final_latent,my_image),0)
        if l==0:
            final_stored = final_latent
        else:
            final_stored = torch.cat((final_stored,final_latent),0)
    return final_stored



def unconditional_sampling(config_path,private_config,ckptpath,private_ckpt,netpath, model_batch_size,total_num_of_samples,output_path):
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
    net = torch.load(netpath)
    data_configs = [OmegaConf.load(config_path)][0]
    dataset_config = data_configs.data
    dataset = instantiate_from_config(dataset_config)
    dataset.prepare_data()
    dataset.setup()
    train_data = DatasetWrapper(dataset.datasets["validation"])
    #z_loader = torch.utils.data.DataLoader(train_data, batch_size=model_batch_size, shuffle=False, drop_last=True)
    zloader = torch.utils.data.DataLoader(train_data, batch_size=model_batch_size,
                                          shuffle=False, num_workers=2)
    i = 0
    for images,label in zloader:
        encoder_posterior = model.encode_first_stage(images.cuda())
        z = model.get_first_stage_encoding(encoder_posterior).detach()
        my_noise = torch.randn_like(z)
        timesteps = torch.randint(800, 801, (z.shape[0],)).long()
        x_noisy = model.q_sample(z.cuda(),timesteps.cuda(),my_noise.cuda()).detach().cpu()
    
        condition = []
        for j in range (images.size(0)):
            condition.append(None)
        condition = torch.vstack(condition)
    
        recovered_samples, recovered_intermediates = sampler.ddim_sampling(condition, size,x_T=x_noisy.cuda(),
                                                        callback=None,
                                                        img_callback=None,
                                                        quantize_denoised=False,
                                                        mask=None, x0=None,
                                                        timesteps = None,
                                                        log_every_t=1,steps_needed=80,break_point=1
                                                        )
        
        new_recovered = recovered_intermediates["pred_x0"][1]
        
        if i == 0:
            latents = new_recovered
        else:
            latents = torch.cat((latents,new_recovered),0)
        i += 1
        
    net.eval()
    with torch.no_grad():
        feature_bank, out_bank = net(latents.cuda())
    private_configs = [OmegaConf.load(private_config)][0]
    private_model = load_model_from_config(private_configs, private_ckpt)
    private_sampler = DDIMSampler(private_model)
    private_sampler.make_schedule(ddim_num_steps=100, ddim_eta=1, verbose=False)
    repeat = int(total_num_of_samples/model_batch_size)
    sampled_dataset = unlabeled_sampling(model_batch_size,repeat,model,sampler,private_sampler,net,feature_bank,train_data)
    torch.save(sampled_dataset,output_path)


unconditional_sampling(config_path,private_config,ckptpath,private_ckpt,model_batch_size,total_num_of_samples,output_path)
def main(args):
    sampling(args.config,args.private_config, args.ckpt, args.private_ckpt, args.netpath, args.batch_size,args.num_samples,args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Path to the public model config file")
    parser.add_argument("--private_config", type=str, required=True, help="Path to the private model config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--private_ckpt", type=str, required=True, help="Path to the private model checkpoint")
    parser.add_argument("--netpath", type=str, required=True, help="Path to the Feature Extractor")
    parser.add_argument("--output", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=50000, help="Total number of images to be sampled")
    args = parser.parse_args()
    main(args)