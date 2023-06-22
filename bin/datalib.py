import os
import sys
import cv2
import torch
import numpy as np

from torchvision import transforms
from PIL import Image


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, imagePathList, classNameList, imageSize, batchSize, grayScale):
        self.imagePathList = imagePathList
        self.classNameList = classNameList
        self.grayScale = grayScale
        self.imageSize = imageSize
        self.batchSize = batchSize

        if self.grayScale == 1:
            self.transform = transforms.Compose([
                transforms.Resize((imageSize, imageSize)),
                transforms.Grayscale(num_output_channels=int(grayScale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((imageSize, imageSize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.imagePathList)

    def __getitem__(self, idx):
        imagePath = self.imagePathList[idx]
        rootPath, file = os.path.split(imagePath)
        fileName, _ = os.path.splitext(file)

        maskPath = os.path.join(rootPath, "MASK_" + fileName + ".png")

        image = cv2.imread(imagePath)

        if self.grayScale == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = Image.open(maskPath).convert("L")
       
        mask = mask.resize((self.imageSize, self.imageSize), resample=0)

        image = Image.fromarray(image)
        image = self.transform(image)
        
        if int(self.batchSize) == 1:
            image = image.squeeze(0)
        mask = np.array(mask)

        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask
    
if __name__ == "__main__":
    trainImageList = ["/data/sungmin/deeplabv3/sample/img/cat.1.jpg"]
    classNameList = ["background", "cat", "dog"]
    imageSize = 512
    batchSize =1
    grayScale = 1
    model = CustomImageDataset(trainImageList, classNameList, imageSize, batchSize, grayScale)
    idx = 0
    model.__getitem__(idx)