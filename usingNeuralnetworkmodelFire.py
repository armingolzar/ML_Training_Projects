import cv2
import glob
import numpy as np
from tensorflow.keras.models import load_model

classifier = load_model('Neuralmodelfire.h5')
output_label = ['fire', 'non fire']
for address in glob.glob('test_fire/*'):
    img = cv2.imread(address)
    re_img = cv2.resize(img, (32, 32))
    re_img = re_img/255.0
    # re_img = re_img.flatten()

    output_pred = classifier.predict(np.array([re_img]))[0]
    max_init = np.argmax(output_pred)
    output = output_label[max_init]

    if output == 'fire':
        cv2.putText(img, 'fire', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    else:
        cv2.putText(img, 'non fire', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)


    cv2.imshow('img', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
