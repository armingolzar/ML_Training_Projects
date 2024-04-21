import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import models, layers


epochs = 20
bach_size = 32

def load_data():
    allData = []
    allLabels = []

    LE = LabelEncoder()

    for index, address in enumerate(glob.glob('fire_dataset/*/*')):
        img = cv2.imread(address)
        img = cv2.resize(img, (32, 32))
        img = img/255.0
        # imgFlatten = img.flatten()

        allData.append(img)

        label = address.split('/')[1]
        allLabels.append(label)

        if index%100 == 0 :
            print('[INFO]: {}/1000 processed'.format(index))

    allData = np.array(allData)


    dataTrain, dataTest, labelTrain, labelTest = train_test_split(allData, allLabels, test_size=0.2, random_state=42)

    labelTrain = LE.fit_transform(labelTrain)
    labelTest = LE.transform(labelTest)

    labelTrain = to_categorical(labelTrain)
    labelTest = to_categorical(labelTest)

    return dataTrain, dataTest, labelTrain, labelTest

def Neural_network():
    net = models.Sequential([
                            layers.Flatten(input_shape=(32, 32, 3)),
                            layers.Dense (30, activation='relu'),
                            layers.Dense(8, activation='relu'),
                            layers.Dense(2, activation='softmax')
                            ])

    net.summary()

    net.compile(optimizer='SGD', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])


    H = net.fit(dataTrain, labelTrain, batch_size=bach_size, epochs=epochs, validation_data=(dataTest, labelTest))
    loss, acc = net.evaluate(dataTest, labelTest)
    print('loss: {:.2f} , acc: {:.2f}'.format(loss, acc))

    net.save('Neuralmodelfire.h5')

    return H

def show_result():
    plt.plot(np.arange(epochs), H.history['loss'], label='train_loss')
    plt.plot(np.arange(epochs), H.history['val_loss'], label='test_loss')
    plt.plot(np.arange(epochs), H.history['accuracy'], label='train_accuracy')
    plt.plot(np.arange(epochs), H.history['val_accuracy'], label='test_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('training on fire dataset')
    plt.show()


dataTrain, dataTest, labelTrain, labelTest = load_data()
H = Neural_network()
show_result()

