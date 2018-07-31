import cv2
import numpy as np

image = cv2.imread('../digits.png')
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