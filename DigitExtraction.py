import cv2
from tensorflow.keras import models, layers

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
    cv2.rectangle(img, (xaxis-5, yaxsis-5), (xaxis+width+5, yaxsis+height+5), (0, 0, 255), 2)



cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()