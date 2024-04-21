import cv2
from tensorflow.keras.models import load_model
import numpy as np

net = load_model('digits.h5')

img = cv2.imread('captcha code.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

conts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


validConts = []

for cont in conts:
    x = cont.shape[0]
    y = cont.shape[1]
    z = cont.shape[2]
    if x*y*z < 50:
        continue

    else:
        validConts.append(cont)



for valCont in validConts:
    xaxis, yaxsis, width, height = cv2.boundingRect(valCont)
    roi = img[yaxsis-5:yaxsis+height+5, xaxis-5:xaxis+width+5]
    roi1 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi1 = cv2.resize(roi1, (32, 32)).flatten()
    roi1 = roi1/255.0
    roi1 = np.array([roi1])

    out = net.predict(roi1)[0]
    # print(out)
    max_index = np.argmax(out) + 1
    # print(max_index)
    cv2.rectangle(img, (xaxis-5, yaxsis-5), (xaxis+width+5, yaxsis+height+5), (0, 0, 255), 2)
    cv2.putText(img, str(max_index), (xaxis-10, yaxsis-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()





