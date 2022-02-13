import json
import random

import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from PIL import Image

from core.taming.models import vqgan
from core.optimizer import DiffGrad, AdamP, RAdam


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


def get_optimizer(z, optimizer="Adam", step_size=0.1):
    if optimizer == "Adam":
        opt = optim.Adam([z], lr=step_size)     # LR=0.1 (Default)
    elif optimizer == "AdamW":
        opt = optim.AdamW([z], lr=step_size)    # LR=0.2
    elif optimizer == "Adagrad":
        opt = optim.Adagrad([z], lr=step_size)  # LR=0.5+
    elif optimizer == "Adamax":
        opt = optim.Adamax([z], lr=step_size)   # LR=0.5+?
    elif optimizer == "DiffGrad":
        opt = DiffGrad([z], lr=step_size)       # LR=2+?
    elif optimizer == "AdamP":
        opt = AdamP([z], lr=step_size)          # LR=2+?
    elif optimizer == "RAdam":
        opt = RAdam([z], lr=step_size)          # LR=2+?
    return opt


def get_scheduler(optimizer, max_iterations, nwarm_restarts=-1):
    if nwarm_restarts == -1:
        return None

    T_0 = max_iterations
    if nwarm_restarts > 0:
        T_0 = int(np.ceil(max_iterations / nwarm_restarts))

    return CosineAnnealingWarmRestarts(optimizer, T_0=T_0)


def load_vqgan_model(config_path, checkpoint_path, model_dir=None):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = vqgan.VQModel(model_dir=model_dir, **config["params"])
    print('1')
    model.eval().requires_grad_(False)
    print(checkpoint_path)

    model.init_from_ckpt(checkpoint_path)
    print('3')

    del model.loss
    return model


def global_seed(seed: int):
    seed = seed if seed != -1 else torch.seed()
    if seed > 2**32 - 1:
        seed = seed >> 32

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to {seed}.")
