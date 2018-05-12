import os
import numpy as np
import cv2
from scipy import stats as sp

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3d plotting을 위해서


all_data_dir = ['./result/test/ruptured/mark/',
                './result/test/unruptured/mark/',
                './result/train/ruptured/mark/',
                './result/train/unruptured/mark/']


def tracking():
    img1 = cv2.imread('./data/test/ruptured/mark/m03913745_1.jpg')
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 99])
    upper_white = np.array([0, 0, 100])

    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    res = cv2.bitwise_and(img1, img1, mask=mask_white)

    cv2.imshow('original', img1)
    cv2.imshow('WHITE', res)

    cv2.waitKey(0)
    cv2.destroyAllWindow()


def check_pixel(arr, x, y):
    pixel1 = (x-1, y)
    pixel2 = (x+1, y)
    pixel3 = (x, y+1)
    pixel4 = (x, y-1)

    if arr[pixel1[1]][pixel1[0]] and arr[pixel2[1]][pixel2[0]] and arr[pixel3[1]][pixel3[0]] and arr[pixel4[1]][pixel4[0]] == 255:
        return True
    else:
        return False


def find_circle(file_name):
    img_color = cv2.imread(file_name, cv2.IMREAD_COLOR)

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    h = img_gray.shape[0]
    w = img_gray.shape[1]

    whitePixelList = []
    whiteXPixelList = []
    whiteYPixelList = []

    white_flag = 0
    threshold = 50
    sensitivity = 2
    # loop over the image, pixel by pixel
    for y in range(threshold, h-threshold):
        for x in range(threshold, w-threshold):
            # threshold the pixel
            if check_pixel(img_gray, x, y):
                white_flag += 1
                if white_flag >= sensitivity:
                    whitePixelList.append((x, y))
                    whiteXPixelList.append(x)
                    whiteYPixelList.append(y)
            else:
                white_flag = 0

    print(whitePixelList)

    centerX = 0
    centerY = 0

    if len(whitePixelList) == 0:
        print("없음")
    else:
        centerX = int((max(whiteXPixelList) + min(whiteXPixelList))/2)
        centerY = int((max(whiteYPixelList) + min(whiteYPixelList))/2)
        print("%d, %d" % (centerX, centerY))
        make_gaussian_array(img_gray, centerX, centerY)



    r = 40

    cv2.circle(img_gray, (centerX, centerY), r, (0, 0, 255), 2)

    cv2.imwrite(file_name, img_gray)

    return (centerX, centerY)


def make_gaussian_array(img, x, y):

    h = img.shape[0]
    w = img.shape[1]

    mu = [x, y]
    cov = [[1024, 0], [0, 1024]]
    rv = sp.multivariate_normal(mu, cov)
    xx = np.linspace(0, w, 255)
    yy = np.linspace(0, h, 255)
    XX, YY = np.meshgrid(xx, yy)
    print(np.dstack([XX, YY]))
    plt.grid(False)
    plt.contourf(XX, YY, rv.pdf(np.dstack([XX, YY])))
    plt.axis("equal")
    plt.show()


def gaussian_test():
    mu = [2, 3]
    cov = [[1, 0], [0, 1]]
    rv = sp.multivariate_normal(mu, cov)
    xx = np.linspace(0, 4, 120)
    yy = np.linspace(1, 5, 150)
    XX, YY = np.meshgrid(xx, yy)
    print(np.dstack([XX, YY]))
    plt.grid(False)
    plt.contourf(XX, YY, rv.pdf(np.dstack([XX, YY])))
    plt.axis("equal")
    plt.show()


def hough_circle():
    print('시작')
    img1 = cv2.imread('C:\Sources\PycharmProjects\Capstone1\data\m00021718_r1.jpg')
    img2 = img1.copy()

    img2 = cv2.GaussianBlur(img2, (3, 3), 0)
    imgray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=70, param2=50, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            cv2.circle(img1, (i[0], i[1]), i[2], (255, 255, 0) ,1)

        cv2.imshow('HoughCircles', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindow()

    else:
        print('원을 찾지 못했습니다.')


#make labeled image
# def onMouse(event,x,y,flags,param):
#     global centerX, centerY;
#     if cv2.EVENT_LBUTTONDOWN == event:
#         centerX = x
#         centerY = y
#
# cv2.namedWindow('win1', cv2.IMREAD_GRAYSCALE)
# cv2.setMouseCallback('win1',onMouse)
# for path_dir in all_data_dir:
#     file_list = os.listdir(path_dir)
#
#     if '.DS_Store' in file_list:
#         file_list.remove('.DS_Store')
#
#     for filename in file_list:
#         input_filename = path_dir+'test/'+filename
#         print(input_filename)
#
#         if (filename != 'Thumbs.db'):
#             img_curr = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#
#             cv2.imshow('win1', img_curr)
#
#
#
#             cv2.circle(img_curr, (centerX, centerY), r, (0, 0, 255), 2)
#
#             cv2.imwrite(input_filename, img_gray)

#gaussian_test()
find_circle('./result/test/unruptured/mark/m00091273_rotate_0.jpg')

# for path_dir in all_data_dir:
#     file_list = os.listdir(path_dir)
#
#     if '.DS_Store' in file_list:
#         file_list.remove('.DS_Store')
#
#     for filename in file_list:
#         input_filename = path_dir+filename
#         print(input_filename)
#
#         if (filename != 'Thumbs.db'):
#            find_circle(input_filename)