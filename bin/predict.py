import os
import sys
import cv2
import json
import time
import torch
import traceback
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf) 
from PIL import Image
from loguru import logger
from torchvision import transforms


basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.extend([basePath, os.path.join(basePath, "model")])

from model import createModel
from logger import Logger
from checker import gpuChecker, initGpuChecker

gpuNo = initGpuChecker()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuNo


class Predictor():
    def __init__(self, pathInfo):

        self.pathInfo = pathInfo
        self.modelPath = self.pathInfo["modelPath"] if "modelPath" in self.pathInfo else '/app'
        self.weightPath = self.pathInfo["weightPath"] if "weightPath" in self.pathInfo else "/app/weight"
        self.log = Logger(logPath=os.path.join(self.modelPath, "log/predict.log"), logLevel="info")

        # set cpu/gpu
        self.setGpu()

        if os.path.isfile(os.path.join(self.weightPath, "weight.pth")):
            with open(os.path.join(self.weightPath, "classes.json"), "r") as jsonFile:
                self.classesJson = json.load(jsonFile)

            self.classNameList = [classInfo["className"] for classInfo in self.classesJson["classInfo"]]
            self.imgSize = self.classesJson["imageInfo"]["imageSize"] if "imageSize" in self.classesJson["imageInfo"] else 512
            self.grayScale = int(self.classesJson["imageInfo"]["imageChannel"])

            if self.grayScale == 1:
                self.transform = transforms.Compose([
                    transforms.Resize((self.imgSize, self.imgSize)),
                    transforms.Grayscale(num_output_channels=self.grayScale),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5)
                ])

            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.imgSize, self.imgSize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            # model load
            logger.info("Model Loading ...")

            modelLoadStartTime = time.time()
            self.model = createModel(
                pretrained=False,
                channel=self.grayScale,
                numClasses=len(self.classNameList)
            )

            self.model.load_state_dict(torch.load(os.path.join(self.weightPath, "weight.pth"), map_location=self.device))
            self.model.eval()
            self.model.to(self.device)

            modelLoadTime = time.time() - modelLoadStartTime
            logger.debug(f"Model Load Success, Duration : {round(modelLoadTime, 4)} sec")

        else:
            raise Exception("This Model is not Trained Model, Not Found Model's Weight File")

    def setGpu(self):
        self.device, self.deviceType = gpuChecker(log=self.log, gpuIdx="auto")

    def runPredict(self, predImage):

        try:
            logger.info("Starting Model Predict...")
            logger.info("-"*100)
            logger.info("  Device:             {}  ".format(self.device.type))
            logger.info("  Image Scaling:      {}  ".format((self.imgSize, self.imgSize, self.grayScale)))
            logger.info("  Labels:             {}  ".format(self.classNameList))

            totalStartTime = time.time()

            # 이미지 예측을 위한 전처리
            logger.info("Input Data Preprocessing for Model...")
            preProStartTime = time.time()

            result = []
            heatMapImage = None
            originImage = predImage.copy()
            height, width = originImage.shape[:2]

            if self.grayScale == 1:
                predImage = cv2.cvtColor(predImage, cv2.COLOR_BGR2GRAY)
            else:
                predImage = cv2.cvtColor(predImage, cv2.COLOR_BGR2RGB)

            predImage = Image.fromarray(predImage)
            predImage = self.transform(predImage)
            #predImage.unsqueeze(0)으로 차원을 한차원 추가한다.
            predImage = predImage.unsqueeze(0)
            predImage = predImage.to(self.device)
   
            preProTime = time.time() - preProStartTime
            logger.debug(f"Input Data Preprocessing Success, Duration : {round(preProTime, 4)} sec")

            # 이미지 예측시작
            logger.info("Predict Start...")

            predStartTime = time.time()
            with torch.no_grad():
                predict = self.model(predImage)['out']

            predTime = time.time() - predStartTime
            logger.debug(f"Predict Success, Duration : {round(predTime, 4)} sec")

            # 예측 결과 형태 변환
            transferOutputStartTime = time.time()
            logger.info("Output Format Transfer...")


            # 예측 결과 마스크 변환 및, 컨투어 좌표값 적용
            mask = predict[0].argmax(0).cpu().numpy().astype(np.uint8)

            predictMaskLabel = list(np.unique(mask))

            logger.debug(f"Predict MaskLabel Info: {predictMaskLabel}")
            del predictMaskLabel[0]

            fullMask = []

            # mask 변환
            for i in predictMaskLabel:
                labelMask = np.where(mask != i, 0, mask)
                labelMask = np.where(labelMask == i, 255, labelMask)

                fullMask.append(labelMask)
             
            
            
            # get segmentation points
            for idx, _labelMask in enumerate(fullMask):

                _labelMask = cv2.resize(_labelMask, (width, height))
                cv2.imwrite("./mask_{}.jpg".format(idx), _labelMask)
                # 윤곽선 검출,contours는 외곽선 좌표 저장, _로 무시처리함(사용하지 않기때문),
                contours, _ = cv2.findContours(_labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cidx, contour in enumerate(contours):
                    contourX = []
                    contourY = []
                    position = []

                    for pt in contour:
                        contourX.append(float(pt[0][0]))
                        contourY.append(float(pt[0][1]))

                    for i in range(len(contourX)):
                        newX1 = max(min(contourX[i], width), 0)
                        newY1 = max(min(contourY[i], height), 0)

                        position.append({
                            "x": newX1,
                            "y": newY1
                        })

                    tmpResult = {
                        "className": self.classNameList[int(predictMaskLabel[idx])],
                        "accuracy": None,
                        "cursor": 'isPolygon',
                        "needCount": -1,
                        "position": position
                    }
                    result.append(tmpResult)
                
                logger.debug(f"Predict className: [{tmpResult['className']}], total: {cidx+1}")

            trasferTime = time.time() - transferOutputStartTime
            logger.debug(f"Output Format Transfer Success, Duration : {round(trasferTime, 4)} sec")

            totalTime = time.time() - totalStartTime
            logger.info(f"Finish Model Predict, Duration : {round(totalTime, 4)} sec")
            logger.info("-"*100)

        except Exception as e:
            logger.error(f"Error :{str(e)}")
            logger.error(f"Traceback : {str(traceback.format_exc())}")

        return result, heatMapImage


if __name__ == "__main__":
    pathInfo = {
        "modelPath": "/data/sungmin/deeplabv3",
        "weightPath": "/data/sungmin/deeplabv3/weight"
    }

    path = "/data/sungmin/deeplabv3/sample/img/cat.1.jpg"

    img = cv2.imread(path)
    p = Predictor(pathInfo)

    # while True:
    predResult, heatMapImage = p.runPredict(img)
