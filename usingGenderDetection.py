import glob
import cv2
from mtcnn import MTCNN
from joblib import load
import numpy as np

detector = MTCNN()
classifier = load('Gender_Classifier.h5')

testing_data = []

def Face_Detector(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detectFace = detector.detect_faces(img)[0]
        x, y, w, h = detectFace['box']
        face = img[y:y+h, x:x+w]
        return face, x, y, w, h

    except:
        pass



for address in glob.glob('./testGender/*'):
    img = cv2.imread(address)
    face, x, y, w, h= Face_Detector(img)
    face = cv2.resize(face, (32, 32))
    face = face/255.0
    face = face.flatten()
    face = face.reshape(1, -1)
    text = classifier.predict(face)[0]
    if text == 'male':
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
    # testing_data.append(face)
   
    


# testing_data = np.array(testing_data)
# for flatten in testing_data:
    # flatten = flatten.reshape(1, -1)
    
    # cv2.putText(img, )



    
