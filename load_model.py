import os
from loadModel import load_model_from_config
import torch
from omegaconf import OmegaConf

def model(models_path,custom_checkpoint_path):
    model_config = "v1-inference.yaml" #@param ["custom","v1-inference.yaml"]
    half_precision = True # check

    # config path
    ckpt_config_path = os.path.join(models_path, model_config)
    if os.path.exists(ckpt_config_path):
        print(f"{ckpt_config_path} exists")
    else:
        ckpt_config_path = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    print(f"Using config: {ckpt_config_path}")

    # checkpoint path or download
    ckpt_path = custom_checkpoint_path
    print(f"Using ckpt: {ckpt_path}")
    local_config = OmegaConf.load(f"{ckpt_config_path}")
    return load_model_from_config(local_config, f"{ckpt_path}", half_precision=half_precision)