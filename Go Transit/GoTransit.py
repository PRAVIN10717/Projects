import cv2
import numpy as np
from tracker import *

# Initialize Tracker
tracker = EuclideanDistTracker()

# Detection confidence threshold
confThreshold = 0.1
nmsThreshold = 0.2

# Attributes for Text
count_font_color = (0, 0, 0)
count_font_size = 0.5
count_font_thickness = 1

font_color = (0, 0, 255)
font_size = 0.30
font_thickness = 1

# In-Out cross line position
in_out_line_position = 460
out_line_position = in_out_line_position - 10
in_line_position = in_out_line_position + 10

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
# Class index of Person
required_class_index = [0]

# configure the network model
modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3-320.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Configure the network backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)

# List to track people information
temp_out_list = []
temp_in_list = []
out_count = 0
in_count = 0

cap = cv2.VideoCapture('Video/People.mp4')
def realTime():
    global cap
    while cap.isOpened():
        success, img = cap.read()
        # Reduce the image shape to 75%
        img = cv2.resize(img,(0,0),None,0.75,0.75)
        ih, iw, channels = img.shape

        input_size = 320
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]

        # Feed data to the network
        outputs = net.forward(outputNames)

        # Find the objects from the network output
        postProcess(outputs, img)

        # Draw the crossing lines
        cv2.line(img, (0, in_out_line_position), (iw, in_out_line_position), (255, 0, 255), 1)
        cv2.line(img, (0, out_line_position), (iw, out_line_position), (0, 0, 255), 1)
        cv2.line(img, (0, in_line_position), (iw, in_line_position), (0, 0, 255), 1)

        # Draw counting texts in the frame
        cv2.putText(img, "In", (70, 20), cv2.FONT_HERSHEY_SIMPLEX, count_font_size, count_font_color, count_font_thickness)
        cv2.putText(img, "Out", (90, 20), cv2.FONT_HERSHEY_SIMPLEX, count_font_size, count_font_color, count_font_thickness)
        cv2.putText(img, "People: "+str(in_count)+"  "+ str(out_count), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, count_font_size, count_font_color, count_font_thickness)

        # Show the frames
        cv2.imshow('Go Transit', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def postProcess(outputs, img):
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

        #color = [0, 0, 255]
        #name = classNames[classIds[i]]
        # Draw classname and confidence score
        #cv2.putText(img, f'{name.upper()} {int(confidence_scores[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
        # Draw bounding rectangle
        #cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

        detection.append([x, y, w, h, required_class_index.index(classIds[i])])
    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)

def count_vehicle(box_id, img):
    global out_count, in_count
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    # Find the current position of person
    if (iy > out_line_position) and (iy < in_out_line_position):
        if id not in temp_out_list:
            temp_out_list.append(id)
    elif (iy < in_line_position) and (iy > in_out_line_position):
        if id not in temp_in_list:
            temp_in_list.append(id)
    elif iy < out_line_position:
        if id in temp_in_list:
            temp_in_list.remove(id)
            out_count = out_count + 1
    elif iy > in_line_position:
        if id in temp_out_list:
            temp_out_list.remove(id)
            in_count = in_count + 1

    # Draw circle in the middle of the rectangle
    #cv2.putText(img, f"{id} : {index} : {iy}", center, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.circle(img, center, 2, font_color, -1)

def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

if __name__ == '__main__':
    realTime()
