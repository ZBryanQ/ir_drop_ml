import os
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

target_size = (930, 930)

# NOTE: load image first, then transform, instead of transforming all possible images at once?

# Turning all the images into a custom Dataset object
image_dataset = torchvision.datasets.ImageFolder(root = '/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/png-files')
print(image_dataset)

# Custom Padding Transformer (FIX THIS)
class PadToSize():
    def init(self, target_size):
        self.target_size = target_size

    def call(self, img):
        # Calculate padding sizes
        width, height = img.size
        pad_width = max(0, self.target_size[0] - width)
        pad_height = max(0, self.target_size[1] - height)

        padding = (0, 0, pad_width, pad_height) # TODO: FIX HEIGHT & WIDTH CALC
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant') # TODO: Change to reflection
    
# Padding Transform Pipeline
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    PadToSize(target_size),  # Pad the image to the target size
    transforms.ToPILImage()  # Convert back to PIL image (optional)
])

image = Image.open('/home/bqtx/Documents/VLSI/ir_drop_ml/training_data/png-files/current/current_map00_current.png')
transformed_image = transform(image)
transformed_image.show()
#torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')