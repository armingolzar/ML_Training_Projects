import cv2
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import models, layers


epochs = 200
bach_size = 32

def load_house_attributes(inputpath):

    cols = ['bedrooms', 'bathrooms', 'area', 'zipcode', 'price']


    df = pd.read_csv(inputpath, sep=' ', header=None, names=cols)

    zipcodes = df['zipcode'].value_counts().keys().to_list()
    counts = df['zipcode'].value_counts().to_list()

    for zipcode, count in zip(zipcodes, counts):

        if count < 25:
            idxs = df[df['zipcode']==zipcode].index

            df.drop(idxs, inplace=True)

    return df


def preprocess_hous_atrribute(df, train, test):
    continues = ['bedrooms', 'bathrooms', 'area']
    sc = StandardScaler()
    traincontinues = sc.fit_transform(train[continues])
    testcontinues = sc.fit_transform(test[continues])


    # encoder = OneHotEncoder(sparse=False)
    # trainCategorical = encoder.fit_transform(np.array(train['zipcode']).reshape(-1, 1))
    # testCategorical = encoder.fit_transform(np.array(test['zipcode']).reshape(-1, 1))

    zipBinarizer = LabelBinarizer().fit(df['zipcode'])

    trainCategorical = zipBinarizer.transform(train['zipcode'])
    testCategorical = zipBinarizer.transform(test['zipcode'])

    trainX = np.hstack([trainCategorical, traincontinues])
    testX = np.hstack([testCategorical, testcontinues])

    return trainX, testX



def Neural_network(dataTrain, dataTest, labelTrain, labelTest):
    net = models.Sequential([
                            layers.Dense (30, activation='relu'),
                            layers.Dense(8, activation='relu'),
                            layers.Dense(1, activation='linear')
                            ])


    net.compile(optimizer='SGD', 
                loss='mse'
               )


    H = net.fit(dataTrain, labelTrain, batch_size=bach_size, epochs=epochs, validation_data=(dataTest, labelTest))
    loss= net.evaluate(dataTest, labelTest)
    print('loss: {:.2f}'.format(loss))

    net.save('Neuralmodelfire.h5')

    return net




df = load_house_attributes('HousesInfo.txt')

train, test = train_test_split(df, test_size=0.2, random_state=42)

trainX , testX = preprocess_hous_atrribute(df , train, test)


max_price = train['price'].max()
trainY = train['price']/max_price
testY = test['price']/max_price


model = Neural_network(trainX, testX, trainY, testY)

preds = model.predict(testX)
diff = preds.flatten() - testY
percentDiff = np.abs((diff/testY)*100)
mean = np.mean(percentDiff)
std = np.std(percentDiff)
print(f'mean: {mean}, std: {std}')
