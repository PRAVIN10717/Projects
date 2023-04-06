import cv2
import numpy
import numpy as np

import handModule as HM
import time
import math

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
print(volume.GetVolumeRange())

handDetector = HM.HandDetector(detection_conf=0.7)

cap = cv2.VideoCapture(0)
pTime = time.time()
volBar = 600
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img, handsFound = handDetector.findHands(img)
    #use handsFound
    for i in range(handsFound):
        lmList = handDetector.findPosition(img, draw=False, handNo=i)
        if len(lmList):
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = math.hypot(x1 - x2, y1 - y2)
            volBar = np.interp(length, [50,450], [600,150])
            #Range of output device
            volLevel = np.interp(length, [50,450], [0,100])
            cv2.putText(img, f'{round(volLevel)}%', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            volume.SetMasterVolumeLevel(volLevel, None)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
            if length > 50:
                cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
            else:
                cv2.circle(img, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (90, 600), (255, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (90, 600), (255, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = time.time()
    cv2.putText(img, f'{int(fps)}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
