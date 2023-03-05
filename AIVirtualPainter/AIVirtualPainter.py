import cv2
import numpy as np
import time
import os
import HandTracking.HandTrackingModule as htm

folderPath = 'Header'
myList = os.listdir(folderPath)
# print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# ####################################3
header = overlayList[0]
drawColor = (255, 0, 255)
brushThickness = 15
eraserThickness = 12
# create new window
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
######################################
cap = cv2.VideoCapture(0)
# set size of window
cap.set(3, 1280)
cap.set(4, 820)
detect = htm.handDetector(detectionCon=0.85)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # Find Hand Landmarks these two functions defined in module
    img = detect.findHands(img)
    lmList = detect.findPosition(img, draw=False)

    if len(lmList) != 0:
        # get index and middle fingure top landmarks
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # check which fingers are up. function defined in module
        fingers = detect.FingerUp()
        # print(fingers)

        # if two fingures are up so it will be selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print('selection mode')
            # area of different colors and eraser
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 130, 220)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                          drawColor, cv2.FILLED)

        # check index fingure
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print('Drawing mode')
            # start where user click first the
            # start of painting
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
                # for eraser
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            # for color drawing
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            xp, yp = x1, y1
    """this portion of code first we convert our image to gray then convert it to image invert image inverse show the background white and object black then cv2.bitwise_end ftn adding two images this code will show the color in black color then at the end we have called cv2.bitwise_or ftn which replace the black color by specific color.
    We have written this code avoid the transperency of sencond image when we drawing color """

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # in the treshold ftn 50 set any value bellow 50 will replace 0 and any value bellow 50 will replace by 255
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    # Covert image from gray to bgr
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)


    imgInv = cv2.resize(imgInv,(1280,720))
    img = cv2.resize(img,(1280,720))
    imgCanvas = cv2.resize(imgCanvas,(1280,720))
    print(img.shape)
    print(imgInv.shape)
    print(imgCanvas.shape)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting header image
    header_resized = cv2.resize(header, (1280, 200))
    h, w, c = header_resized.shape
    print(h,w)
    img[0:h, 0:w] = header_resized
    # set two images in single window
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow('Image', img)
    cv2.imshow('Canvas', imgCanvas)
    # cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# cv2.distroyAllwindows()
