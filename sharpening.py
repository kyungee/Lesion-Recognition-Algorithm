import os, sys
import cv2
import numpy as np
import scipy.ndimage as ndi

all_data_dir = ['./original/test/ruptured/', './original/train/ruptured/',
                './original/test/unruptured/', './original/train/unruptured/',
                './original/test/normal/', './original/train/normal/']


def crop_img(_input, w, h):
    start = abs(int((w-h)/2))

    if w > h:
        cropped = _input[0:h, start:start+h]
    else:
        cropped = _input[start:start+w, 0:w]

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

        img = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)

        sharpen = np.array((
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]), dtype="int")

        sh_img = cv2.filter2D(img, -1, sharpen)

        f_filename = filename.split('.jpg')[0]
        output_filename = f_filename+'_sharpening_'+str(filename)+'.jpg'

        path = r'./result/sharpening/%s%s' % (path_dir[11:], output_filename)
        print(path)
        cv2.imwrite(path, sh_img)


print("\n * Completed!!!!")

