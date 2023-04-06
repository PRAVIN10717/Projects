import mediapipe as mp
import cv2
import time

class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, tracking_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = 1
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf
        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_conf, self.tracking_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        self.foundHand = True if self.result.multi_hand_landmarks else False
        self.handsFound = 0
        if self.foundHand:
            self.handsFound = len(self.result.multi_hand_landmarks)
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHand.HAND_CONNECTIONS)
        return img, self.handsFound

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.foundHand:
            handLms = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(w * lm.x), int(h * lm.y)
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                lmList.append([id, cx, cy])
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime = time.time()
    cTime = pTime
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img, handsFound = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList):
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()