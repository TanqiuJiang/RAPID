# RAPID: Retrieval Augmented Training of Differentially Private Diffusion Models

This is the offical implemation of [RAPID: Retrieval Augmented Training of Differentially Private Diffusion Models](https://openreview.net/forum?id=txZVQRc2ab). \
The code is based off a public implementation of Latent Diffusion Models, available [here](https://github.com/CompVis/latent-diffusion) and a public implementation of [Differentially Private Latent Diffusion Models](https://openreview.net/pdf?id=FLOxzCa6DS) available [here](https://github.com/SaiyueLyu/DP-LDM).

# Environment setup:

```sh
conda env create -f environment.yaml
conda activate RAPID
```

# Model Training

## Training the Autoencoder
```bash
python main.py --base <AE config file path> -t --gpus 1
```

## Training the Diffusion Models
```bash
python main.py --base <DM config file path> -t --gpus 1
```

## Private Model Finetuning

```bash
python main.py --base <Finetune config file path> -t --gpus 0,  --accelerator gpu
```

## Feature Extractor Training
```bash
python train_feature_extractor.py --config <DM config file path> --ckpt <checkpoint path> --output <network output path> --epoch 50
```

# Sampling 
## Conditional Sampling
```bash
python conditional_sampling.py --config <DM config file path> --private_config <DM config file path> --ckpt <checkpoint path> \
 --private_ckpt <checkpoint path> --netpath <path to the feature extractor> --output <network output path> 

``` 

## Unconditional Sampling
```bash
python unconditional_sampling.py --config <DM config file path> --private_config <DM config file path> --ckpt <checkpoint path> \
 --private_ckpt <checkpoint path> --netpath <path to the feature extractor> --output <network output path> 
``` 

# FID Evaluation
```bash
python FID_test.py --sample_path <path to generated samples> --train_stats_path <path to generated statistics on the reference set>
``` 

# Diversity Evaluation
```bash
python Diversity_test.py --sample_path <path to generated samples> --data_config <config file path>
``` 

# Downstream Classification Accuracy
For MNIST, to compute the downstream performance on a regular CNN, the command is:
```bash
CNN_downstream.py --sample_path <path to generated samples> --epoch 10
```


# Acknowledgement
We built and tested our project on top of [Differentially Private Latent Diffusion Models](https://openreview.net/pdf?id=FLOxzCa6DS) and [Differentially Private Latent Diffusion Models](https://openreview.net/pdf?id=FLOxzCa6DS). Many thanks to the authors who make their work publicly accessable! 

