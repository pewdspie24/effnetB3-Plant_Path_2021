import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import shutil
# import tqdm
import tqdm.notebook as tq #for colab
from utils import Model, get_device, accuracy
import utils
import torch.optim as optim
from sklearn.model_selection import KFold
from dataset import CustomDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = CustomDataset('./data')
k_folds = 5
kfold = KFold(n_splits=5, shuffle=True)
torch.manual_seed(2)
batch_size = 16
LR = 1e-3
epochs = 6 #total epoch = 6*5 = 30
checkpoint_path = './checks'
model = Model().cuda()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, cooldown=1)

losses_train = []
losses_valid = []
accuracy_train = []
accuracy_valid = []
monitor_acc = 0.0
name = 'tmp'

for fold,(train_idx,val_idx) in enumerate(kfold.split(dataset)):
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)
    trainLoader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=2)
    valLoader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=2)

    for epoch in range(epochs):
        if epoch == (epochs//2):
            model.unfreeze()
            print("Start finetuning")
        running_train_loss = 0.0
        running_valid_loss = 0.0
        running_train_accuracy = 0.0
        running_valid_accuracy = 0.0
        model.train()
        for (img,label) in tq.tqdm(trainLoader):
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            # print(img.size())
            output = model(img) 
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * img.size(0)
            running_train_accuracy += accuracy(output, label) 
        epoch_train_loss = running_train_loss / len(trainLoader)
        epoch_train_acc = running_train_accuracy / len(trainLoader)
        losses_train.append(epoch_train_loss)
        accuracy_train.append(epoch_train_acc)
        print("")
        print('Training, Epoch {} - Loss {} - Acc {}'.format(epoch+1, epoch_train_loss, epoch_train_acc))

        model.eval()
        with torch.no_grad():
            for (img,label) in tqdm.tqdm(valLoader):
                img = img.to(device)
                output = model(input) 
                loss = criterion(output, label)
                running_valid_loss += loss.item() * img.size(0) 
                running_valid_accuracy += accuracy(output, label)
            epoch_valid_loss = running_valid_loss / len(valLoader)
            epoch_valid_acc = running_valid_accuracy / len(valLoader)
            losses_valid.append(epoch_valid_loss)
            accuracy_valid.append(epoch_valid_acc)
            print("")
            print('Validating, Epoch {} - Loss {} - Acc {}'.format(epoch+1, epoch_valid_loss, epoch_valid_acc))
        if(epoch_valid_acc > monitor_acc):
            monitor_acc = epoch_valid_acc
            name = f"EfficientNetB3_epoch-{epoch+1+fold*5}_acc-{epoch_valid_acc}.pth"
            os.remove('./checks/*.pth')
            model.save(model.state_dict(), os.path.join(checkpoint_path,name))

plt.plot(losses_train, label="train_loss")
plt.plot(losses_valid, label="valid_loss")
plt.plot(accuracy_train, label="train_acc")
plt.plot(accuracy_valid, label="valid_acc")  
plt.legend(loc="upper left")
#if you want to save it to drive when train on colab
shutil.copy(os.path.join(checkpoint_path,name),os.path.join("/content/drive/MyDrive/AI-ML/Plant_Pathology/",name))

