import cv2
import numpy as np
import pandas as pd

def preprocess_image(image_dirname):
    image_path = 'cv2/images/' + image_dirname
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 去照光
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    # 手动选择电表的四个角点（根据实际情况调整）
    points = np.array([[100, 200], [400, 200], [400, 500], [100, 500]], dtype='float32')
    width, height = 300, 300
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(points, dst_points)

    # 进行透视变换
    warped = cv2.warpPerspective(adaptive_thresh, matrix, (width, height))

    return warped

df = pd.read_csv('cv/images/标签.csv')

# 使用示例
processed_image = preprocess_image('hefei_3188.jpg')

# 显示结果
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()