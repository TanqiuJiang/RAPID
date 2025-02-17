import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from scipy import linalg
from pytorch_fid.inception import InceptionV3
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from prdc import compute_prdc

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

def get_features(my_dataloader,mymodel):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pred_list = []
        pbar = tqdm(my_dataloader)
        for batch in pbar:
            x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
            x = x.to(device)

            pbar.set_description(f"x.shape={tuple(x.shape)} | x.min()={x.min()} | x.max()={x.max()}")

            with torch.no_grad():
                pred = mymodel(x)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = torch.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_list.append(pred)

        pred_arr = np.concatenate(pred_list, axis=0)
        return pred_arr

class myDataset(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
    
    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        item = torch.as_tensor(item["image"])
        item = item.permute(2, 0, 1)
        return item

def diversity_test(sample_path,data_config_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inception_model = InceptionV3(normalize_input=False).to(device)
    
    new_dataset = torch.load(sample_path)
    
    data_configs = [OmegaConf.load(data_config_path)][0]
    dataset_config = data_configs.data
    dataset = instantiate_from_config(dataset_config)
    dataset.prepare_data()
    dataset.setup()
    
    test_data = myDataset(dataset.datasets["validation"])
    new_loader = DataLoader(test_data, batch_size=500, shuffle=False, drop_last=True)
    
    real_feature =  get_features(new_loader,inception_model)

    mydataset = DatasetWrapper(new_dataset)
    dataloader = DataLoader(dataset=mydataset, batch_size=500)
    
    fake_feature =  get_features(dataloader,inception_model)
    
    metrics = compute_prdc(real_features=real_feature[0:5000],
                           fake_features=fake_feature[0:5000],
                           nearest_k=5)
    
    print(metrics)

def main(args):
    diversity_test(args.sample_path,args.data_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sample_path", type=str, required=True, help="Path to the generated sample")
    parser.add_argument("--data_config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    main(args)
    
