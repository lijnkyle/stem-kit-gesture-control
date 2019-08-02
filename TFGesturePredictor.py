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

AI_win = True

start_gesture = True

# global variables
bg = None

datagen = ImageDataGenerator(rescale = 1./255)

def resizeImage(imageName):
    img = Image.open(imageName)
    img = img.resize((150,150),Image.ANTIALIAS)
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


def extract_features(directory, sample_count):
    features=np.zeros(shape=(sample_count, 4, 4, 512))
    labels=np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(directory,target_size=(150,150),batch_size=1,class_mode='categorical')
    for inputs_batch, labels_batch in generator:
        features_batch=pre_model.predict(inputs_batch)
        features[0]=features_batch
        labels[0]=labels_batch
    return features, labels

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

#            cv2.imwrite('Temp.png', roi)
            tempdir = os.path.join('real-time-image','x')
            cv2.imwrite(os.path.join(tempdir, 'Temp.png'),roi)
            
            resizeImage('Temp.png')
            predictedClass, confidence = getPredictedClass()
            showStatistics(predictedClass, confidence, AI_win)
                    #start_recording = False
#time.sleep(3)
#                cv2.imshow("Thesholded", thresholded)

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


cwd = os.path.dirname(__file__)
abs_file_path = os.path.join(cwd,'real-time-image')
print(abs_file_path)


def getPredictedClass():
    # Predict
    image = cv2.imread(os.path.join('real-time-image', 'Temp.png'))
    pre_features, pre_labels = extract_features(abs_file_path, 1)
    pre_features = np.reshape(pre_features, (1,4*4*512))
    pre_labels = to_categorical(pre_labels)
    print(pre_labels)
    prediction = model.predict(pre_features)
    print(prediction)
    return np.amax(prediction),0
    #prediction = model.predict([image.reshape(150, 150, 1)])
#return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))

def showStatistics(predictedClass, confidence, AI_win):

    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0 and confidence > 0.90:
        className = "Rock"
        
        if _USE_ARDUINO:


            if AI_win:
                ser.write(paper)
            else:
                ser.write(scissor)
            ser.close



    elif predictedClass == 1 and confidence > 0.90:
        className = "Paper"
        if _USE_ARDUINO:

            if AI_win:
                ser.write(scissor)
            else:
                ser.write(rock)
            ser.close

    elif predictedClass == 2 and confidence > 0.90:
        className = "Scissor"
        if _USE_ARDUINO:

            if AI_win:
                ser.write(rock)
            else:
                ser.write(paper)
            ser.close

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
model = keras.models.load_model('TrainedModel/GestureRecogModel.h5')
model.summary()
main()

