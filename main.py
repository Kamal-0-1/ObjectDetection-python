import cv2
import matplotlib.pyplot as plt

config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn.DetectionModel(frozen_model, config)
# classLabels=[]
file_name = 'labels.txt'
with open(file_name, 'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127, 5, 127.5))
model.setInputSwapRB(True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open the video')
font_scale = 3
font = cv2.FONT_HERSHEY_DUPLEX
while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    print(classLabels[ClassIndex[0] - 1])
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 255, 0))
    cv2.imshow('Object', frame)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
