from flask import Flask, render_template, Response
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

app=Flask('Mask Detection')

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab dimensions of frame then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,1.0, (192,192),
                                (104.0,177.0,123.0))
    
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)
    
    # init list of faces and corresponding locations, and the list of predictions from our facemask network
    
    faces = []
    locations = []
    predictions = []
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence(probability) associated with the detection
        confidence = detections[0,0,i,2]
        
        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype('int')
            
            # make sure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))
            
            # extract the face Region Of Interest, convert it from BGR to RGB channel ordering, resize it, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224,224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locations.append((startX, startY, endX, endY))
            
    # only make the predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on all faces at the same time rather than one-by-one predictions in the obove for loop
        faces = np.array(faces)
        predictions = maskNet.predict(faces, batch_size=32)
        
    # return tuple of the face locations and their corresponding locations
    return (locations, predictions)

# load our serialized face detector model from disk
prototxt_path = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxt_path,weightsPath)

# load face mask detector model from disk
maskNet = load_model("detect_mask.model")

def gen_frames():
    while True:
        success, frame = vs.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=400)
    
            # detect faces in the frame and determ if they are wearing mask, not wearing, or worn incorrectly
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
            # loop over detected fadce locations and their corresponding locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
        
                # determine the class label and color we'll use to draw the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
                # incl probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
                # display the label and bounding box rectangle on the output frame
                cv2.putText(frame,label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX,0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            ret, buffer = cv2.imencode('.png', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

# init the video stream
vs = VideoStream(0).start()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
