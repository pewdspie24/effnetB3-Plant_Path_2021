import torch
import glob
import cv2
import os
from random import shuffle
from torch.utils.data import DataLoader
from torchvision import transforms
import dataset
from sklearn.model_selection import KFold

# labels = ['complex', 
#           'frog_eye_leaf_spot',
#           'frog_eye_leaf_spot complex',
#           'healthy',
#           'powdery_mildew',
#           'powdery_mildew complex',
#           'rust',
#           'rust complex',
#           'rust frog_eye_leaf_spot',
#           'scab',
#           'scab frog_eye_leaf_spot',
#           'scab frog_eye_leaf_spot complex'
#           ]
def get_dataset(path):
    k_folds = 5 #20% / fold
    dataset = dataset.CustomDataset(path, "./train.csv")
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
