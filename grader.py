import cv2
import numpy as np
import imutils
from imutils import contours    
import four_point

Answer_kkey= {0:1, 1:4, 2:0, 3:3, 4:1}

image = cv2.imread("omr.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)
cnts= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts= imutils.grab_contours(cnts)
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri= cv2.arcLength(c, True)
        approx= cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            docCnt= approx
            break

paper= four_point.four_point_transform(image, docCnt.reshape(4, 2))
warped= four_point.four_point_transform(gray, docCnt.reshape(4, 2))
thresh= cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts= imutils.grab_contours(cnts)
questionCnts= []

for c in cnts:
    (x, y, w, h)= cv2.boundingRect(c)
    ar= w/float(h)
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

questionCnts= contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct= 0

for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    cnts= contours.sort_contours(questionCnts[i:i+5])[0]
    bubbled= None
    for (j, c) in enumerate(cnts):
        mask= np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask= cv2.bitwise_and(thresh, thresh, mask=mask)
        total= cv2.countNonZero(mask)
        if bubbled is None or total > bubbled[0]:
            bubbled= (total, j)
    color= (0, 0, 255)
    k= Answer_kkey[q]
    if k == bubbled[1]:
        color= (0, 255, 0)
        correct += 1
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
print(correct)

