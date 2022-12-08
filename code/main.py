import cv2
import numpy as numpy
import lbph as lr
import sys, os
from time import sleep
import msvcrt
import csv
from gtts import gTTS
from playsound import playsound

from train import TrainFromSavedPhotos,TrainFromWebcam

savedModelLocation = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorP\\code\\trained_face_model.npy'
baseDir = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\res\\data'
fn_haar = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\code\\haarcascade_frontalface_default.xml'
fn_dir = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\res\\database'


def play__sound(s):
    mytext = s
    language = 'en-us'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    filename = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\testvoice.mp3'
    myobj.save(filename)
    playsound(filename)
    os.remove('C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\testvoice.mp3')



def LoadModelAndRun():
    trained_face_recognizer=numpy.load(savedModelLocation)
    # Load prebuilt model for Frontal Face
    cascadePath = "C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\code\\haarcascade_frontalface_default.xml"
    (im_width, im_height) = (68, 68)
    # Part 2: Use fisherRecognizer on camera stream
    (images, lables, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(fn_dir):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(fn_dir, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                lable = id
                images.append(cv2.imread(path, 0))
                lables.append(int(lable)) 
            id += 1

    face_cascade = cv2.CascadeClassifier(cascadePath)
    webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))
            prediction=lr.predict_lbph(face_resize,trained_face_recognizer,lables)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            if (prediction[1])<=100 and (prediction[1])>85:
                print('%s - %s' % (names[prediction[0]],"marked PRESENT"))
                cv2.putText(im,'%s - %.0f%s' % (names[prediction[0]],prediction[1],"%"),(x-10, y-10), cv2.FONT_HERSHEY_TRIPLEX,2,(0, 255, 0))
                s = str(names[prediction[0]]) + "marked PRESENT"
                play__sound(s)
                sleep(2)
                play__sound("Next Student, Please Come Forward")
                # space
                # for
                # handling enrollment number of recognized face to send request to DB
            else:
                cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    cv2.destroyAllWindows()


while True:
        print('\n___________________________________________\n\n0.Train from Saved Photos\n1.Train From Webcam \n2.Run\n3.Exit: \n\n')
        play__sound("Choose 0 to train from Saved Photos, Choose 1 to train from Webcam, Choose 2 to Run Recognition and Choose 3 to Exit.")
        user=input("-->")
        if user == '0':
            TrainFromSavedPhotos()
            
        elif user == '1':
            TrainFromWebcam()
            
        elif user=='2':
            LoadModelAndRun()
    
        elif user == '3':
            print("Exiting!")
            break
        else:
            print("Enter Valid input!\n\n")
            play__sound("Please Enter valid Input")