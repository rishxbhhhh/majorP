import cv2
import numpy as numpy
import lbph as lr
import os
from time import sleep
from gtts import gTTS
from playsound import playsound
from datetime import date,datetime
import pandas as pd
from train import TrainFromSavedPhotos,TrainFromWebcam
recognized__students__list = list()
date_today_compressed = date.today().strftime("%a-%d-%m-%y")
date_today_detailed = date.today().strftime("%a-%d-%B-%Y")

attendance__dir = "C:\\Users\\Rishabh Rajpurohit\\Documents\\majorP\\Attendance"

sub__name = "default subject"
cls__type = "lecture"
faculty__name = "default faculty"
cls__time = "9am"
s__no = 1

if f'Attendance-{date_today_detailed}.csv' not in os.listdir(attendance__dir):
    with open(f'Attendance/Attendance-{date_today_detailed}.csv','w') as f:
        f.write('S-No,Enrollment-No,Time-Stamp(Hour:Min:Sec),Subject-Name,Class-Type,Faculty-Name,Class-Time')

from train import TrainFromSavedPhotos,TrainFromWebcam

curr_dir = os.path.dirname(__file__)

savedModelLocation = os.path.join(curr_dir, 'trained_face_model.npy')
baseDir = os.path.join(curr_dir, '..', 'res', 'data')
fn_haar = (curr_dir, 'haarcascade_frontalface_default.xml')
fn_dir = (curr_dir, '..', 'res', 'database')

# savedModelLocation = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorP\\code\\trained_face_model.npy'
# baseDir = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\res\\data'
# fn_haar = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\code\\haarcascade_frontalface_default.xml'
# fn_dir = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\res\\database'


sound__file = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\testvoice.mp3'
def play__sound(s):
    mytext = s
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    # filename = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\testvoice.mp3'
    filename = os.path.join(curr_dir, '..', 'testvoice.mp3')
    myobj.save(filename)
    playsound(filename)
    os.remove(filename)
    # os.remove('C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\testvoice.mp3')



def LoadModelAndRun():
    s__no = 1
    recognized__students__list.clear()
    sub__name, cls__type, faculty__name, cls__time = input('''\nEnter the following Details:\nFormat: <Sub_name>,<Class_type>,<Faculty_Name>,<Class_time>\n--->''').split(',')
    try:
        trained_face_recognizer=numpy.load(savedModelLocation)
    except:
        print('\n\nTrain the model first!')
        return
    # Load prebuilt model for Frontal Face
    # cascadePath = "C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\code\\haarcascade_frontalface_default.xml"
    cascadePath = fn_haar
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

    face_cascade = cv2.CascadeClassifier(fn_haar)
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
                current__time = datetime.now().strftime('%H:%M:%S')
                df = pd.read_csv(f'Attendance/Attendance-{date_today_detailed}.csv')
                if(recognized__students__list.count(names[prediction[0]])==0):
                    recognized__students__list.append(names[prediction[0]])
                    with open(f'Attendance/Attendance-{date_today_detailed}.csv', 'a') as f:
                        f.write(f'\n{s__no},{names[prediction[0]]},{current__time},{sub__name},{cls__type},{faculty__name},{cls__time}')
                        s__no += 1
                    print('%s - %s' % (names[prediction[0]],"marked PRESENT"))
                    cv2.putText(im,'%s - %.0f%s' % (names[prediction[0]],prediction[1],"%"),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                    play__sound(str(names[prediction[0]]) + "marked PRESENT")
                    #play__sound("next student, please come forward")
                
            else:
                cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

        #cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    cv2.destroyAllWindows()


def init():
    while True:
        #play__sound("Choose 0 to train from Saved Photos, Choose 1 to train from Webcam, Choose 2 to Run Recognition and Choose 3 to Exit.")
        print('___________________________________________\n\n0.Train from Saved Photos\n1.Train From Webcam \n2.Run \n3.Reset Model \n4.Exit:\n')
        user=input("-->")
        if user == '0':
            TrainFromSavedPhotos()
            
        elif user == '1':
            TrainFromWebcam()
            
        elif user=='2':
            try:
                LoadModelAndRun()
            except:
                break
            finally:
                init()
    
        elif user=='3':
            os.remove(savedModelLocation)

        elif user == '4':
            print("Exiting!")
            break
        else:
            print("Enter Valid input!\n\n")
            play__sound("Please Enter valid Input")
