import cv2
import pandas as pd

df = pd.read_csv('cv/images/标签.csv')
image_dirname = 'hefei_3188.jpg'
image_path = '/cv2/images/' + image_dirname

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Processed Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()