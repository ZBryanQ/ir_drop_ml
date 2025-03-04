import os
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from mobilenetv3 import MobileNetV3_Small
from PIL import Image

# NOTE: load image first, then transform, instead of transforming all possible images at once?


# Custom Padding Transformer (FIX THIS)
class PadToSize:
    def __init__(self, target):
        self.target = target

    def __call__(self, img):
        # Calculate padding sizes
        print(img.size())
        colors, width, height = img.size()
        pad_left = (max(0, self.target[0] - width)//2)
        pad_right = ((max(0, self.target[0]-width)+1)//2)
        pad_top = (max(0, self.target[1] - height)//2)
        pad_bot = ((max(0,self.target[1] - height)+1)//2)

        padding = (pad_left, pad_top, pad_right, pad_bot)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='reflect')
    
# Padding Transform Pipeline
target_size = (930, 930)
pipeline= transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Lambda(lambda x: x[:3]),
    PadToSize(target_size),  # Pad the image to the target size
    # transforms.ToPILImage(), 
])

# Turning all the images into a custom Dataset object
image_dataset = torchvision.datasets.ImageFolder(root = '/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/png-files', transform = pipeline)
#print(image_dataset)

image = Image.open('/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/png-files/current/current_map00_current.png')
# temp = transforms.ToTensor()
# print(temp(image))
transformed_image = pipeline(image).size()
print(transformed_image)
#transformed_image.show()
#torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')

network = MobileNetV3_Small(2)
# print(network)

# random = torch.rand(2,3,224,224)
network(transformed_image)