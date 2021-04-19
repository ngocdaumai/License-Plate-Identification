import cv2
import numpy as np
import pytesseract
import numpy as np
from PIL import ImageGrab
import time

from skimage.segmentation import clear_border
pytesseract.pytesseract.tesseract_cmd = 'E:\\Program Files\\Tesseract-OCR\\tesseract.exe'
cap = cv2.VideoCapture(0)
whT = 320

confThreshold =0.5
nmsThreshold= 0.5
classesFile = "coco_plate.names"
# classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = "yolov4-tiny_custom.cfg"
modelWeights = "yolov4-tiny_custom_plate.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
path = "E:/Daumn/1.PythonCode/Computervision/train_data/images/train/"

def build_tesseract_options(psm=7):
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    options += " --psm {}".format(psm)
    return options


def findObjects(outputs, img):
    img_copy = img.copy()
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(  det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        if (0< x) & (y > 0) & (h > 0) & (w>0):
            imgForName = img_copy[y:y+h, x:x+w]
            imgGray = cv2.cvtColor(imgForName, cv2.COLOR_BGR2GRAY)
            roi = cv2.threshold(imgGray, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            roi = clear_border(roi)
            cv2.imshow("roi", roi)
            options = build_tesseract_options(psm=7)
            lpText = pytesseract.image_to_string(roi, config=options)
            print(lpText)
            cv2.putText(img, lpText,
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()
    # img = cv2.imread("plate_test12.jpg")
    img = cv2.resize(img, (640, 460))
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)
    cv2.imshow("original", img)
    if cv2.waitKey(1)==ord("q"):
        break

cv2.destroyAllWindows()
