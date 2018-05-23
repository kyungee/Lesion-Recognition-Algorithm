import cv2


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