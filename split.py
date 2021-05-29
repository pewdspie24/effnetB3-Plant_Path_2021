import os
import shutil
import glob

#split for ur life, joking, just because data is too much for colab so i need to spilt it
images_path = './data/train_images'
size = 9000
i = 0
for image in glob.glob(images_path+'/*.jpg'):
    name = image.split('/')[-1]
    shutil.move(image, './train_images/'+name)
    i+=1
    if i > size:
        break