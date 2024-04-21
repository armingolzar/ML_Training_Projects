import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical  
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt



def load_data():


    data = []
    labels = []

    for index, address in enumerate(glob.glob('kapcha/*/*')):

        img = cv2.imread(address)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32)).flatten()
        img = img/255.0
    

        data.append(img)
        
    

        label = address.split('/')[-2]
        labels.append(label)

        if index % 100 == 0:
            print(f'[INFO]: {index}/2000 processed')

    data = np.array(data)

    dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, labels, test_size=0.2, random_state=42)


    LE = LabelEncoder()

    labelTrain = LE.fit_transform(labelTrain)
    labelTest = LE.transform(labelTest)

    labelTrain = to_categorical(labelTrain)
    labelTest = to_categorical(labelTest)

    return dataTrain, dataTest, labelTrain, labelTest 

def Training():

    net = models.Sequential([layers.Dense(64, activation='relu'),
                            layers.Dense(32, activation='relu'),
                            layers.Dense(9, activation='softmax')
                            ])


    net.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )


    H = net.fit(dataTrain, labelTrain, batch_size=32, epochs=20, validation_data=(dataTest, labelTest))

    net.save('digits.h5')

    return H

def show_result():
    plt.style.use('ggplot')
    plt.plot(H.history['accuracy'], label='train accuracy')
    plt.plot(H.history['val_accuracy'], label='test accuracy')
    plt.plot(H.history['loss'], label='train loss')
    plt.plot(H.history['val_loss'], label='test loss')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/loss')
    plt.legend()
    plt.show()


dataTrain, dataTest, labelTrain, labelTest = load_data()
H = Training()
show_result()
