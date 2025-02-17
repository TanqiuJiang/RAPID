import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from scipy import linalg
from fid.cifar10_fid_stats_pytorch_fid import stats_from_dataloader, set_seeds
from pytorch_fid.inception import InceptionV3
import numpy as np

class DatasetWrapper(Dataset):
    def __init__(self, new_dataset):
        self.images = new_dataset

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        image = self.images[i]
        assert image.shape[0] == 3 and image.shape[1] == image.shape[2], \
               f"Samples not in CxHxW format, instead got {image.shape}"
        image = image.clamp(min=-1, max=1)
        return image

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    m = np.square(mu1 - mu2).sum()
    s, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    fd = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    return fd
    
def FIDtest(sample_path,train_stats_path):
    new_dataset = torch.load(sample_path) 
    mydataset = DatasetWrapper(new_dataset)
    dataloader = DataLoader(dataset=mydataset, batch_size=100)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inception_model = InceptionV3(normalize_input=False).to(device)
    mu, sigma = stats_from_dataloader(dataloader, inception_model, device)
    stats1 = np.load(train_stats_path)
    stats1_mu = stats1['mu']
    stats1_sigma = stats1['sigma']
    fid = calculate_frechet_distance(stats1_mu, stats1_sigma, mu, sigma)
    print('FID: %.4f' % fid)
    
def main(args):
    FIDtest(args.sample_path,args.train_stats_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sample_path", type=str, required=True, help="Path to the generated sample")
    parser.add_argument("--train_stats_path", type=str, required=True, help="Path to the generated train stats")
    args = parser.parse_args()
    main(args)
    