#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%pip install matplotlib
#%pip install imageio


# In[2]:


#import IPython
import os
import json

import torch

from core.schemas import Config

from scripts.generate import *

import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import imageio

from tqdm import tqdm

#get_ipython().run_line_magic('matplotlib', 'notebook')
#reload_model = True


# In[3]:


MYPATH = './samples/freddieart/'
CONFIG_FILE = './configs/local.json'
DEVICE = torch.device(os.environ.get("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu')) #'cpu'
def cap_space(text):
    new_text = ''
    for i, letter in enumerate(text):
        if i and letter.isupper():
            new_text += ' '
        new_text += letter
    return new_text


# In[4]:


# clean text file names
image_files = [f for f in os.listdir(MYPATH) if os.path.isfile(os.path.join(MYPATH, f))]


# In[5]:


cleaned_file_names = []
for file_name in image_files:
    #remove parens
    file_name = file_name.replace('(', ' ')
    file_name = file_name.replace(')', ' ')
    
    if '+' in file_name:
        #replace + with space
        file_name = file_name.replace('+', ' ')
    elif '_' in file_name:
        file_name = file_name.replace('_', ' ')
    elif ' ' in file_name:
        pass
    else:
        #in files without white space, add space in front of capital letters
        file_name = cap_space(file_name)
    file_name = file_name.replace('  ', ' ')
    file_name = file_name.replace('.jpg', '')
    file_name = file_name.rstrip().lstrip()
    cleaned_file_names.append(file_name)
    


# In[6]:


with open(CONFIG_FILE, 'r') as f:
    PARAMS = Config(**json.load(f))
model = load_vqgan_model(PARAMS.vqgan_config, PARAMS.vqgan_checkpoint, PARAMS.models_dir).to(DEVICE)
perceptor = clip.load(PARAMS.clip_model, device=DEVICE, root=PARAMS.models_dir)[0].eval().requires_grad_(False).to(DEVICE)
cut_size = perceptor.visual.input_resolution
make_cutouts = MakeCutouts(PARAMS.augments, cut_size, PARAMS.cutn, cut_pow=PARAMS.cut_pow)

z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


# In[7]:


def initialize_params(prompt='a painting'):
    #print(f"Loading default configuration from '{CONFIG_FILE}'")
    with open(CONFIG_FILE, 'r') as f:
        PARAMS = Config(**json.load(f))
    PARAMS.prompts = [prompt]
    PARAMS.init_noise = 'fractal'
    #PARAMS.init_image = './samples/VanGogh.jpg'
    #print(f"Running on {DEVICE}.")
    #print(PARAMS)

    global_seed(PARAMS.seed)

    z = initialize_image(model, PARAMS)
    z_orig = torch.zeros_like(z)
    z.requires_grad_(True)

    prompts = tokenize(model, perceptor, make_cutouts, PARAMS)
    optimizer = get_optimizer(z, PARAMS.optimizer, PARAMS.step_size)
    scheduler = get_scheduler(optimizer, PARAMS.max_iterations, PARAMS.nwarm_restarts)

    kwargs = {
        'model': model,
        'perceptor': perceptor,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'prompts': prompts,
        'make_cutouts': make_cutouts,
        'z_orig': z_orig,
        'z_min': z_min,
        'z_max': z_max,
        'mse_weight': PARAMS.init_weight,
    }
    return PARAMS, kwargs, z


# In[8]:


#fig,ax = plt.subplots(1,1)

for file_index in tqdm(range(len(cleaned_file_names))):    
    PARAMS, kwargs, z = initialize_params(prompt=cleaned_file_names[file_index])
    output_filename = '_'.join(PARAMS.prompts).replace(' ', '_')
    if os.path.exists(f"{PARAMS.output_dir}/{output_filename}.gif"):
        continue

    tqdm.write(f"prompt: {PARAMS.prompts}")
    for step in range(PARAMS.max_iterations):
        kwargs['step'] = step + 1
        pil_image = train(z, PARAMS, **kwargs)
 #       if step % 15 == 0:
            #ax.imshow(np.asarray(pil_image))
            #plt.axis('off')
            #fig.canvas.draw()
    
    if len(PARAMS.prompts):
        output_dir = f"{PARAMS.output_dir}/steps/"
        output_image_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        output_image_files = sorted(output_image_files)
        images = []
        for filename in output_image_files:
            images.append(imageio.imread(output_dir+filename)) 
        images.append(pil_image)
        imageio.mimsave(f"{PARAMS.output_dir}/{output_filename}.gif", images, duration=1)


# In[ ]:


#shutdown kernel
#IPython.Application.instance().kernel.do_shutdown(False) 


# In[ ]:


output_dir = f"{PARAMS.output_dir}/steps/"
output_image_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
images = []
for filename in output_image_files:
    images.append(imageio.imread(output_dir+filename))                
output_filename = '_'.join(PARAMS.prompts).replace(' ', '_')
imageio.mimsave(f"{PARAMS.output_dir}/{output_filename}.gif", images, duration=1)


# In[ ]:




