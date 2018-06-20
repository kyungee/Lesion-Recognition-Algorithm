import os
import csv
import math
import numpy as np
import cv2
from cv2 import matchTemplate as cvm
import pprint
from scipy import stats as sp
from matplotlib import pyplot as plt



test_dir = ['./temp/testcase/u/', './temp/testcase/r/']
original_r_data_dir = './temp/testcase/ruptured/'
original_u_data_dir = './temp/testcase/unruptured/'
result_data_file_dir = './temp/testcase/result/'
all_data_dir = ['./temp/testcase/ruptured/', './temp/testcase/unruptured/']


def make_file_list(folder_dir):
    final_nfile_list = []
    final_rfile_list = []
    final_ufile_list = []
    for path_dir in folder_dir:
        file_list = os.listdir(path_dir)

        if '.DS_Store' in file_list:
            file_list.remove('.DS_Store')

        for filename in file_list:
            input_filename = path_dir+filename

            if filename != 'Thumbs.db':
                if path_dir[-2] == 'n':
                    final_nfile_list.append(input_filename)
                if path_dir[-2] == 'r':
                    final_rfile_list.append(input_filename)
                if path_dir[-2] == 'u':
                    final_ufile_list.append(input_filename)

    return (final_nfile_list, final_rfile_list, final_ufile_list)


def get_marked_image(file_dir):
    file_number, file_tag = split_file_name(file_dir)
    original_image = cv2.imread(file_dir)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    w, h = original_image.shape[::-1]
    top_left = find_image_pixel(original_image, file_number, file_tag)
    flag_str = ''
    if file_tag == 'r':
        flag_str = 'ruptured'
    if file_tag == 'u':
        flag_str = 'unruptured'
    img_marked_path = './temp/testcase/' + flag_str + ' - marking/' + file_number + '.jpg'
    img_marked = cv2.imread(img_marked_path)
    img_marked = cv2.cvtColor(img_marked, cv2.COLOR_BGR2GRAY)
    img_trimmed = img_marked[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]
    cv2.imwrite(result_data_file_dir+file_number+file_tag+'.jpg', img_trimmed)
    return img_trimmed


def cv2_based(field_array,match_array):
    M = cvm(field_array.astype('uint8'),match_array.astype('uint8'),cv2.TM_SQDIFF)
    return np.where(M==0)


def split_file_name(file_dir):
    dir_string = file_dir.split('-')
    file_number = dir_string[0][-4:]
    file_tag = dir_string[1][0]
    return (file_number, file_tag)


def find_image_pixel(image, file_number, file_tag):
    flag_str = ''
    if file_tag == 'r':
        flag_str = 'ruptured'
    if file_tag == 'u':
        flag_str = 'unruptured'
    cmp_img_path = './temp/testcase/' + flag_str + '/' + file_number + '.jpg'
    cmp_img = cv2.imread(cmp_img_path)
    cmp_gray_image = cv2.cvtColor(cmp_img, cv2.COLOR_BGR2GRAY)

    w, h = image.shape[::-1]

    methods = ['cv2.TM_SQDIFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        method = eval(meth)

        res = cv2.matchTemplate(cmp_gray_image, image, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        img_marked_path = './temp/testcase/' + flag_str + ' - marking/' + file_number + '.jpg'
        img_marked = cv2.imread(img_marked_path)
        img_marked = cv2.cvtColor(img_marked, cv2.COLOR_BGR2GRAY)
        img_trimmed = img_marked[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

        x, y, size = find_circle(img_trimmed)
        if size is not None:
            return top_left

    return None


def check_pixel(arr, x, y):
    pixel1 = (x-1, y)
    pixel2 = (x+1, y)
    pixel3 = (x, y+1)
    pixel4 = (x, y-1)

    if arr[pixel1[1]][pixel1[0]] and arr[pixel2[1]][pixel2[0]] and arr[pixel3[1]][pixel3[0]] and arr[pixel4[1]][pixel4[0]] == 255:
        return True
    else:
        return False


def find_circle(img_gray):
    h = img_gray.shape[0]
    w = img_gray.shape[1]

    whitePixelList = []
    whiteXPixelList = []
    whiteYPixelList = []

    white_flag = 0
    threshold = 1
    size = 0
    sensitivity = 1
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
                    size += 1
            else:
                white_flag = 0

    x = 0
    y = 0
    if len(whitePixelList) == 0:
        x = None
        y = None
        size = None
    else:
        x = int((max(whiteXPixelList) + min(whiteXPixelList))/2)
        y = int((max(whiteYPixelList) + min(whiteYPixelList))/2)

    return (x, y, size)


def make_gaussian_array(m_x=60, m_y=50, cov_size=128):
    mu = [m_x, m_y]
    cov = [[cov_size, 0], [0, cov_size]]
    rv = sp.multivariate_normal(mu, cov)
    xx = np.linspace(0, 127, 128)
    yy = np.linspace(0, 127, 128)
    XX, YY = np.meshgrid(xx, yy)
    rv_array = rv.pdf(np.dstack([XX, YY]))
    h, w = rv_array.shape
    rv_array = rv_array * 10e25

    rv_ceil_array = np.zeros((h, w))
    for y in range(0, h):
        for x in range(0, w):
            normalized = math.ceil(rv_array[y][x])
            rv_ceil_array[y][x] = math.ceil(normalized)

    minimum = rv_ceil_array.min()
    maximum = rv_ceil_array.max()
    for y in range(0, h):
        for x in range(0, w):
            normalized = (rv_ceil_array[y][x] - minimum) / (maximum - minimum) * 255
            if math.ceil(normalized) == 1.0:
                rv_ceil_array[y][x] = 0.0
            else:
                rv_ceil_array[y][x] = math.ceil(normalized)

    return rv_ceil_array


def gaussian_test(file_dir, m_x=2, m_y=3, cov_size=1):
    xx = np.linspace(0, 127, 128)
    yy = np.linspace(0, 127, 128)
    XX, YY = np.meshgrid(xx, yy)
    plt.grid(False)
    rv_result_array = make_gaussian_array(m_x, m_y, cov_size)

    plt.contourf(XX, YY, rv_result_array)
    plt.axis("equal")

    fig = plt.gcf()
    fig.savefig(file_dir+'.pdf')


def make_nparray_to_csv(file_dir, np_array):
    f = open(file_dir+'.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    h, w = np_array.shape
    for y in range(0, h):
        wr.writerow(np_array[y])
    f.close()


def make_nparray_to_jpg(file_dir, np_array):
    cv2.imwrite(file_dir+'_n.jpg', np_array)


# Main 함수
item_list = []
totalListTuple = make_file_list(test_dir)
for file_dir in totalListTuple[1]:
    file_number, file_tag = split_file_name(file_dir)
    img_trim = get_marked_image(file_dir)
    x, y, size = find_circle(img_trim)
    if size is not None:
        print('x= %d, y= %d, size= %d' % (x, y, size))
        print(file_number + file_tag)
        item_list.append((file_dir, x, y, size))
        gaussian_test(result_data_file_dir + file_number + file_tag, x, y, size)
        make_nparray_to_csv(result_data_file_dir + file_number + file_tag, make_gaussian_array(x, y, size))
        make_nparray_to_jpg(result_data_file_dir + file_number + file_tag, make_gaussian_array(x, y, size))

for file_dir in totalListTuple[2]:
    file_number, file_tag = split_file_name(file_dir)
    img_trim = get_marked_image(file_dir)
    x, y, size = find_circle(img_trim)
    if size is not None:
        print('x= %d, y= %d, size= %d' % (x, y, size))
        print(file_number + file_tag)
        item_list.append((file_dir, x, y, size))
        #gaussian_test(result_data_file_dir + file_number + file_tag, x, y, size)
        #make_nparray_to_csv(result_data_file_dir + file_number + file_tag, make_gaussian_array(x, y, size))
        make_nparray_to_jpg(result_data_file_dir + file_number + file_tag, make_gaussian_array(x, y, size))

# 데이터를 csv로 추출
# f = open('test.csv', 'w', encoding='utf-8', newline='')
# wr = csv.writer(f)
# for item in item_list:
#     wr.writerow(item)
# f.close()
