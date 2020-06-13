import os
import pandas as pd 
import numpy as np 
import tqdm as tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms
import  torchvision.models as models

input_path = 'dataset'
train_image_path = 'resized_images/'
test_image_path = 'resized_images'
train_df = pd.read_csv(os.path.join(input_path,'train.csv'))
test_df = pd.read_csv(os.path.join(input_path,'test.csv'))
final_df = pd.read_csv(os.path.join(input_path,'final_train.csv'))













