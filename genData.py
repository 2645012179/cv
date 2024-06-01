import numpy as np
import cv2
import tensorflow as tf

def preprocess_image(img_path, img_width=230, img_height=32):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 中值滤波去噪
    gray = cv2.medianBlur(gray, 5)

    points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype='float32')
    width = xmax - xmin
    height = ymax - ymin
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')

    # 去照光
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(points, dst_points)
    
    # 进行透视变换
    warped = cv2.warpPerspective(adaptive_thresh, matrix, (width, height))
    
    res_image = cv2.resize(warped,(230,32))
    
    solved_image = np.expand_dims(res_image, axis=-1)
    
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, annotations, batch_size, char_list, imgH, imgW):
        self.annotations = annotations
        self.batch_size = batch_size
        self.char_list = char_list
        self.imgH = imgH
        self.imgW = imgW

    def __len__(self):
        return int(np.floor(len(self.annotations) / self.batch_size))

    def __getitem__(self, index):
        batch_annotations = self.annotations[index * self.batch_size:(index + 1) * self.batch_size]

        X = np.array([preprocess_image(ann['image_path'], self.imgW, self.imgH) for ann in batch_annotations])
        Y = np.array([self.encode_label(ann['label']) for ann in batch_annotations])
        
        input_length = np.ones((self.batch_size, 1)) * (self.imgW // 4 - 2)
        label_length = np.array([[len(ann['label'])] for ann in batch_annotations])

        return [X, Y, input_length, label_length], np.zeros(self.batch_size)

    def encode_label(self, label):
        return [self.char_list.index(char) for char in label]

    def on_epoch_end(self):
        pass
