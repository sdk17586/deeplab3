import os
import json
from loguru import logger
from checker import  pathChecker, datChecker
from torch.utils.data import DataLoader, random_split
from datalib import CustomImageDataset
from maker import makeMaskImage


def createDataLoader(filePathList, classInfo, classNameList, imageSize, batchSize, grayScale, trainRate):
    
    logger.info("createDataLoader")
    imagePathList = pathChecker(filePathList, dataType="image")

    imagePathList = datChecker(imagePathList)

    makeMaskImage(imagePathList, classNameList)

    if len(imagePathList) > 50:
        logger.warning("Dataset size > 50")

        trainSize = len(imagePathList) * float(trainRate["train"] / 100)
        validSize = len(imagePathList) - trainSize

        trainImageList, validImageList = random_split(imagePathList, [int(trainSize), int(validSize)])

        trainDataset = CustomImageDataset(trainImageList, classNameList, imageSize, batchSize, grayScale)
        trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, pin_memory=True)

        validDataset = CustomImageDataset(validImageList, classNameList, imageSize, batchSize, grayScale)
        validDataLoader = DataLoader(validDataset, batch_size=batchSize, shuffle=True, drop_last=True, pin_memory=True)

        return trainDataLoader, validDataLoader

    else:
        logger.warning("Dataset size < 50")
        trainSize = len(imagePathList)
        validSize = 0
        
        trainImgList, validImageList = random_split(imagePathList, [int(trainSize), int(validSize)])

        trainDataset = CustomImageDataset(trainImgList, classNameList, imageSize, batchSize, grayScale)
        trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, pin_memory=True)

        return trainDataLoader, trainDataLoader
    
    
# if __name__ == "__main__":
#     filePathList =
#     classInfo =
#     classNameList = []
#     imageSize = 512
#     batchSize = 1
#     grayScale = 3
#     trainRate = {
#       "train": 70,
#       "validation": 30
#     }
#     createDataLoader(filePathList, classInfo, classNameList, imageSize, batchSize, grayScale, trainRate):