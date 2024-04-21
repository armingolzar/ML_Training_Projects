import cv2
import glob
from mtcnn import MTCNN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from joblib import  dump

detector = MTCNN()
data = []
labels = []



def Face_Detector(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detectFace = detector.detect_faces(img)[0]
        x, y, w, h = detectFace['box']
        face = img[y:y+h, x:x+w]
        return face

    except:
        pass


for index, address in enumerate(glob.glob('./Gender/*/*')):
    img = cv2.imread(address)
    face = Face_Detector(img)


    if face is None:
        continue
    
    else:
        face = cv2.resize(face, (32, 32))
        cv2.imshow('face', face)
        face = face/255.0
        face = face.flatten()
        data.append(face)

        label = address.split('/')[-2]
        labels.append(label)

        if index % 100 == 0:
            print('[INFO]: {}/3300 data processed'.format(index))


data = np.array(data)
dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, labels, test_size=0.2, random_state=42)

classifier = SGDClassifier()

classifier.fit(dataTrain, labelTrain)

prediction = classifier.predict(dataTest)
acc = accuracy_score(labelTest, prediction)
print('accuracy: {:.2f}'.format(acc*100))

dump(classifier, 'Gender_Classifier.h5')

        

    
        

    



    

    

        