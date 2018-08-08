import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while True:
    _, img = cap.read()

    cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
    
    crop_img = img[100:300, 100:300]

    cv2.imshow('Frame', img)
    cv2.imshow('Cropped', crop_img)
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)


    kernel = (35, 35)
    blurred = cv2.GaussianBlur(grey, kernel, 0)

    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)

    image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # drawing contours
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    if cv2.waitKey(0) == 32:
        break

cv2.destroyAllWindows()