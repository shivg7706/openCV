import cv2
import numpy as np

image = cv2.imread('./digits.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(image)

# cv2.imshow('Digits', small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cells = np.array([np.hsplit(row, 100) for row in np.vsplit(gray, 50)])

train_data = cells[:, :70].reshape(-1, 400).astype(np.float32)
test_data = cells[:, 70:].reshape(-1, 400).astype(np.float32)

label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

train_labels = np.repeat(label, 350)[:, np.newaxis]
test_labels = np.repeat(label, 150)[:, np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

_, result, neighbour, distance = knn.findNearest(test_data, k=3)


matches = (result == test_labels)
acc = np.count_nonzero(matches) * (100/result.size)
print('Acccuracy = {}%'.format(acc))
print('Model Trained Successfully')

def x_coor_contour(contour):

    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        return int(M['m10']/M['m00'])


def convert_to_square(img):

    BLACK = [0, 0, 0] 
    height, width = img.shape

    if height == width:
        return img

    else:
        height, width = 2*height, 2*width
        doubleSize = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        if height > width:
            pad = (height - width)//2
        else:
            pad = (width - height)//2
        new_img = cv2.copyMakeBorder(doubleSize, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=BLACK)

    return new_img


def convert_to_pixel(dimension, image):

    dimension -= 4
    tasveer = image
    r = float(dimension) / tasveer.shape[1]
    dim = (dimension, int(tasveer.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    if resized.shape[0] > resized.shape[1]:
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if resized.shape[0] > resized.shape[1]:
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # added border which was taken out before
    img = cv2.copyMakeBorder(resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img

image = cv2.imread('./1.png')
cv2.imshow('Original', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('blurred', blurred)
edged = cv2.Canny(blurred, 30, 150)
cv2.imshow('edged', edged)

_, contour, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour = sorted(contour, key=x_coor_contour)

for c in contour:

    (x, y, w, h)  = cv2.boundingRect(c)

    if w >= 5 and h >= 25:

        roi = blurred[y:y+h, x:x+w]
        _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        square = convert_to_square(roi)
        f_img = convert_to_pixel(20, square).reshape(1, 400).astype(np.float32)
        _, result, neighbour, distance = knn.findNearest(f_img, k=1)

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, '{}'.format(int(result[0])), (x, y+100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

