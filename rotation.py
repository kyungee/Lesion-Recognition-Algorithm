
import os, sys
import cv2


all_data_dir = ['./original/test/Ruptured/', './original/train/Ruptured/',
                './original/test/Unruptured/', './original/train/Unruptured/',
                './original/test/Normal/', './original/train/Normal/',]


def crop_img(input, w, h):
    start = abs(int((w-h)/2))

    if w > h:
        cropped = input[0:h, start:start+h]
    else:
        cropped = input[start:start+w, 0:w]

    return cropped


for path_dir in all_data_dir:
    file_list = os.listdir(path_dir)

    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    if 'Thumbs.db' in file_list:
        file_list.remove('Thumbs.db')

    for filename in file_list:
        input_filename = path_dir+filename
        print(input_filename)

        img = cv2.imread(input_filename, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height, width = img.shape[:2]

        if width != height:
            img = crop_img(img, width, height)

        height, width = img.shape[:2]

        for i in range(1, 4):
            angle = i*90
            M = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1)
            dst = cv2.warpAffine(img, M, (width, height))

            f_filename = filename.split('.jpg')[0]
            output_filename = f_filename+'_rotate_'+str(angle)+'.jpg'

            path = r'./%s%s' % (path_dir[2:], output_filename)
            print(path)
            cv2.imwrite(path, dst)


print("\n * Completed!!!!")

#cv2.imshow('Original', img)
#cv2.imshow('warpAffine', dst)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
