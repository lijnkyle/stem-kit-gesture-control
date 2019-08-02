import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils
from tensorflow import keras
import time
import os

from keras.preprocessing.image import ImageDataGenerator

serial_port_num ='14130'

_USE_ARDUINO = False
if _USE_ARDUINO:
    import serial
    ser = serial.Serial('/dev/cu.usbserial-' + serial_port_num, 9600)

paper = b'0'
rock = b'1'
scissor = b'2'
reset = b'1'

cwd = os.path.dirname(__file__)
img_path = os.path.join(cwd, 'real-time-image/temp.png')

AI_win = True

start_gesture = True

# global variables
bg = None

datagen = ImageDataGenerator(rescale = 1./255)

def resizeImage(imageName):
    img = Image.open(imageName)
    img = img.resize((150,150), Image.ANTIALIAS)
    img.save(imageName)

#def run_avg(image, aWeight):
#    global bg
#    # initialize the background
#    if bg is None:
#        bg = image.copy().astype("float")
#        return
#
#    # compute weighted average, accumulate it and update the background
#    cv2.accumulateWeighted(image, bg, aWeight)

def main():
    
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_EXPOSURE, 0.25)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    start_recording = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        cv2.imshow("roi", roi)

#
#        if num_frames < 30:
#            run_avg(g_channel, aWeight)
#        else:
#            # segment the hand region
#            hand = segment(gray)
#
#
#
            # check whether hand region is segmented
#            if hand is not None:
#                # if yes, unpack the thresholded image and
#                # segmented region
#                (thresholded, segmented) = hand

                    #draw the segmented region and display the frame
#                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
        if start_recording:
            cv2.imwrite(img_path ,roi)
            resizeImage(img_path)
            confidence, predictedClass = getPredictedClass()
            showStatistics(predictedClass, confidence, AI_win)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

        if keypress == ord("v"):
            start_recording = False

        
        if keypress == ord("s"):
            AI_win = True
            start_recording = True
        
        if keypress == ord("z"):
            AI_win = False
            start_recording = True

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

def getPredictedClass():
    # Predict
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    predicted_features = pre_model.predict(preprocess_input(x))
    prediction = model.predict(np.reshape(predicted_features, (1,4*4*512)))
    print(prediction)
    return np.amax(prediction), np.argmax(prediction)

def showStatistics(predictedClass, confidence, AI_win):

    textImage = np.zeros((300,512,3), np.uint8)

    if predictedClass == 0 and confidence > 0.90:
        className = "Rock"

        if _USE_ARDUINO:

            if AI_win:
                ser.write(paper)
            else:
                ser.write(scissor)
            ser.close()



    elif predictedClass == 1 and confidence > 0.90:
        className = "Paper"
        if _USE_ARDUINO:

            if AI_win:
                ser.write(scissor)
            else:
                ser.write(rock)
            ser.close()

    elif predictedClass == 2 and confidence > 0.90:
        className = "Scissor"
        if _USE_ARDUINO:

            if AI_win:
                ser.write(rock)
            else:
                ser.write(paper)
            ser.close()

    else:
        className = "None"


    cv2.putText(textImage,"AI : "+ str(AI_win),
    (30, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2)



    cv2.putText(textImage,"Pedicted Class : " + className,
    (30, 100),
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    (30, 200),
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)



    cv2.imshow("Statistics", textImage)

pre_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(150,150,3))
model = keras.models.load_model(os.path.join(cwd, 'TrainedModel' ,'TLTrainedModel.h5'))
model.summary()
main()

