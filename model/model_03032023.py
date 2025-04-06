import os
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from vit import ViT as vit 
from PIL import Image
import math as math

# NOTE: load image first, then transform, instead of transforming all possible images at once?

# Custom Padding Transformer (FIX THIS)
class PadToSize:
    def __init__(self, target):
        self.target = target

    def __call__(self, img):
        # Calculate padding sizes

        colors, width, height = img.size()
        pad_left = math.floor((max(0, self.target[0] - width)/2))
        pad_right = math.ceil(((max(0, self.target[0]-width))/2))-1
        pad_top = math.floor((max(0, self.target[1] - height)/2))+1
        pad_bot = math.ceil((max(0,self.target[1] - height)/2))

        padding = (pad_left, pad_top, pad_right, pad_bot)
        if self.target[0]-width > width or self.target[1]-height > height:
            return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
        else:
            return transforms.functional.pad(img, padding, fill=0, padding_mode='reflect')
    

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_folder = torchvision.datasets.ImageFolder(root=root_dir, transform=transform)
        self.imgs = self.image_folder.imgs

    def __getitem__(self, index):
        data, target = self.image_folder[index]
        path, _ = self.imgs[index]
        return data, target, path

    def __len__(self):
        return len(self.imgs)

# Padding Transform Pipeline
target_size = (930, 930)
pipeline= transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Lambda(lambda x: x[:3]),
    PadToSize(target_size),  # Pad the image to the target size
    #transforms.ToPILImage(), 
])

# Turning all the images into a custom Dataset object
image_dataset = CustomDataset(root_dir = '/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/png-files', transform = pipeline)
#for image in image_dataset:
#    print(image)
#print(image_dataset)

#image = Image.open('/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/png-files/pdn_density/testcase5_pdn_density.png')
#temp = transforms.ToTensor()
#print(temp(image))
#transformed_images = pipeline(image_dataset)
#print(transformed_image)
#transformed_image.show()
#torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')

#network = MobileNetV3_Small(2)
# print(network)

# random = torch.rand(2,3,224,224)
#network(transformed_image)

#'''
model = vit(image_size = (930,930), patch_size = (15,15), num_classes = 2, dim = 3, depth = 1, heads = 4, mlp_dim = 10) 

random_image = torch.randn(3, 3, 930, 930)

model(random_image)

counter = 0
stacked_img_list = list()
dict_of_lists = {}
# for i, (image, label, path) in enumerate(image_dataset):
#     print(i)
#     #print(image.size())
#     print(path)
#     print(label)
    # if label not in dict_of_lists.keys:
    #     dict_of_lists[label] = list()
    # dict_of_lists[label].append(image)
    # stacked_img_list.append(image)
    # counter+=1
    # if counter > 3:
    #     stacked_img = torch.stack(stacked_img_list)
    #     print(stacked_img.shape)
    #     model(stacked_img)
    #     break
#'''