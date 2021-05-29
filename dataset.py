import torch
import os 
import glob
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.images = []
        self.labels = []

        image_path = glob.glob(data_path+'/train_images/*.jpg')
        label_csv = pd.read_csv(data_path+'/train.csv')

        for path in image_path:
            self.images.append(path)
            self.labels.append(label_csv[label_csv['image']==path.split('/')[-1]].to_numpy()[-1][-1])

        random_apply = [transforms.RandomAffine(degrees=(-45,+45), scale=(1,2))]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply(random_apply, p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        self.labels_list = ['complex', 
          'frog_eye_leaf_spot',
          'frog_eye_leaf_spot complex',
          'healthy',
          'powdery_mildew',
          'powdery_mildew complex',
          'rust',
          'rust complex',
          'rust frog_eye_leaf_spot',
          'scab',
          'scab frog_eye_leaf_spot',
          'scab frog_eye_leaf_spot complex'
        ]
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (300,300))
        label = self.labels_list.index(self.labels[idx])
        image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    dataset = CustomDataset("./data")
    print(dataset.__getitem__(1)[1])
