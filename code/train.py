import os,time
import cv2
import numpy
import lbph as lr
from playsound import playsound
from gtts import gTTS

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
    filename = os.path.join(curr_dir, '..', testvoice.mp3)
    myobj.save(filename)
    playsound(filename)
    os.remove(filename)
    # os.remove('C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\testvoice.mp3')

def TrainFromSavedPhotos():
    persons = os.listdir(baseDir)
    print("Fetching Data...")
    #play__sound('fetching data')
    for person in persons:
        images = os.listdir(baseDir+"\\"+person)
        count = 0
        size = 4
        fn_name = person
        path = os.path.join(fn_dir, fn_name)
        if not os.path.isdir(path):
            os.mkdir(path)
        (im_width, im_height) = (68, 68)
        haar_cascade = cv2.CascadeClassifier(fn_haar)

        for image in images:
            pathOfImg = baseDir+"\\"+person+"\\"+image
            im = cv2.imread(pathOfImg)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            mini = cv2.resize(gray, ((int)(gray.shape[1] / size), (int)(gray.shape[0] / size)))
            faces = haar_cascade.detectMultiScale(mini)
            faces = sorted(faces, key=lambda x: x[3])
            if faces:
                count=count+1
                face_i = faces[0]
                (x, y, w, h) = [v * size for v in face_i]
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))
                pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0]!='.' ]+[0])[-1] + 1
                cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_TRIPLEX,2,(0, 255, 0))
                if(count>=20):
                    break; 

    print('Training...')
    #play__sound('the model is now training')

    # Create a list of images and a list of corresponding names
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
    (im_width, im_height) = (68, 68)

    # Create a Numpy array from the two lists above
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]
    trained_face_recognizer=lr.train_lbph(images)
    print('done')
    #play__sound('the model is now trained')
    numpy.save(savedModelLocation,trained_face_recognizer)


def TrainFromWebcam():
    count = 0
    size = 4
    #play__sound('Please enter your name or enrollment number')
    fn_name = input('Enter Your Enrollment Number: ') 
    path = os.path.join(fn_dir, fn_name)
    if not os.path.isdir(path):
        os.mkdir(path)
    (im_width, im_height) = (68, 68)
    haar_cascade = cv2.CascadeClassifier(fn_haar)
    webcam = cv2.VideoCapture(0)

    print("--------------Ensure that the room is well lit--------------")
    print("-----------------------Taking pictures----------------------")
    #play__sound("Ensure the room is well lit and the distance between face and camera is not more than half a meter for better accuracy")
    # The program loops until it has a few images of the face.

    while count < 20:
        (rval, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(gray, ((int)(gray.shape[1] / size), (int)(gray.shape[0] / size)))
        faces = haar_cascade.detectMultiScale(mini)
        faces = sorted(faces, key=lambda x: x[3])
        if faces:
            face_i = faces[0]
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))
            pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0]!='.' ]+[0])[-1] + 1
            cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_TRIPLEX,2,(0, 255, 0))
            time.sleep(0.38)        
            count += 1
        
            
        #cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    print(str(count) + " images taken and saved to " + fn_name +" folder in database ")
    cv2.destroyAllWindows()
    webcam.release()
    size = 4
    print('Training...')
    #play__sound('the model is now training')
    # Create a list of images and a list of corresponding names
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
    (im_width, im_height) = (68, 68)

    # Create a Numpy array from the two lists above
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]
    trained_face_recognizer=lr.train_lbph(images)
    print('done')
    #play__sound('the model is now trained')
    numpy.save(savedModelLocation,trained_face_recognizer)
