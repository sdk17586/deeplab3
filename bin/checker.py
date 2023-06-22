import os
import gc
import cv2
import glob
import json
import time
import torch
import mimetypes
import GPUtil as gp
from loguru import logger


def initGpuChecker():

    gpus = gp.getGPUs()

    if len(gpus) > 0:
        usableDevices = {str(gpu.id): int(gpu.memoryFree) for gpu in gpus}
        gpuNo = max(usableDevices, key=usableDevices.get)
    else:
        gpuNo = ""

    return gpuNo


def gpuChecker(log=None, gpuIdx="auto", gpuNo=None):

    if gpuIdx == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if str(device) == "cpu":
            deviceType = "cpu"
            device = torch.device("cpu")

            if log is not None:
                log.warning("Cuda is not available...")
                log.info("Set Device: CPU")

        else:
            deviceType = "gpu"
            device = torch.device("cuda")

            if log is not None:
                log.info("Checking gpuIdx auto...")
                log.info("Set Device: GPU:{}".format(gpuNo))

    return device, deviceType


def pathChecker(pathList, type="dir", dataType="image"):

    '''
        정보 1. 데이터 타입이 이미지/비디오인지 체크 / 파일이 열리는지 안열리는지 체크하여 열리는 파일만 리스트 형태로 변환
    '''
 
    logger.info("Checking Path for dataset...")
    pathChekerTime = time.time()

    filterStrList = ["THUM", "MASK_"]
    filePathList = []

    if type == "dir":
        for fileDirPath in pathList:
            # logger.debug(f"Checking fileDirPath : {fileDirPath}")

            # glob 리스트
            globList = glob.glob(os.path.join(fileDirPath, "*.*"))

            # filterStrList, fileType != None, fileType == "image/video" 체크
            filteredList = [globPath for globPath in globList if not any(fstr in globPath for fstr in filterStrList) and
                            mimetypes.guess_type(globPath)[0] is not None and
                            mimetypes.guess_type(globPath)[0].split("/")[0] == dataType]

            # 이미지 파일이 열리는지 체크
            if dataType == "image":
                filePathList.extend([filePath for filePath in filteredList if cv2.imread(filePath) is not None])

            # 비디오 파일이 열리는지 체크
            elif dataType == "video":
                filePathList.extend([filePath for filePath in filteredList if cv2.VideoCapture(filePath).read()[0] is True])

    else:
        # filterStrList, fileType != None, fileType == "image/video" 체크
        filteredList = [path for path in pathList if not any(fstr in pathList for fstr in filterStrList) and
                        mimetypes.guess_type(path)[0] is not None and
                        mimetypes.guess_type(path)[0].split("/")[0] == dataType]

        # 이미지 파일이 열리는지 체크
        if dataType == "image":
            filePathList.extend([filePath for filePath in filteredList if cv2.imread(filePath) is not None])

        # 비디오 파일이 열리는지 체크
        elif dataType == "video":
            filePathList.extend([filePath for filePath in filteredList if cv2.VideoCapture(filePath).read()[0] is True])

    filteredList = None

    gc.collect()
    pathChekerTotalTime = time.time() - pathChekerTime
    logger.info(f"Finish Checking Path for dataset, Duration : {round(pathChekerTotalTime, 4)} sec")
    print(filePathList)
    return filePathList


# .DAT 존재, 내부 정보가 있는 파일만 추출
def datChecker(filePathList):

    '''
        정보 1. .dat 파일이 있고, 내부 정보가 있는 이미지/비디오 파일만 추출해서 리스트 형태로 변환
    '''

    pathList = []

    for filePath in filePathList:
        rootName, fileExtension = os.path.splitext(filePath)
        datPath = rootName + ".dat"

        if os.path.isfile(datPath):
            with open(datPath, "r") as jsonFile:
                datData = json.load(jsonFile)

            if os.stat(datPath).st_size == 0:
                logger.warning(f'{datPath} file is Empty.')
                continue

            elif not datData["polygonData"] and not datData["brushData"]:
                logger.warning(f'{datPath} file has No Label Data.')
                continue

            else:
                pathList.append(filePath)
        else:
            logger.warning(f'{datPath} file not exist.')
   
    return pathList



if __name__ == "__main__":
    filePathList = ['/data/sungmin/deeplabv3/sample/img']
    imagePathList = pathChecker(filePathList, dataType="image")