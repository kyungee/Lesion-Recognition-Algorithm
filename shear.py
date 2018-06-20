import os, sys
import cv2
import numpy as np
import scipy.ndimage as ndi


all_data_dir = ['./original/test/Ruptured/', './original/train/Ruptured/',
                './original/test/Ruptured-Mark/', './original/train/Ruptured-Mark/',
                './original/test/Unruptured/', './original/train/Unruptured/',
                './original/test/Unruptured-Mark/', './original/train/Unruptured-Mark/',
                './original/test/Normal/', './original/train/Normal/',]


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

        img = cv2.imread(input_filename, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        c_in = 0.5 * np.array(img.shape)
        c_out = 0.5 * np.array(img.shape)

        intensity = 20.0

        for i in range(1, 6):
            theta = np.pi/180 * np.random.uniform(-intensity, intensity)
            inv_rotation = np.array([[np.cos(theta), np.sin(theta)], [0, 1]] / np.cos(theta))
            offset = c_in - np.dot(inv_rotation, c_out)
            out = (ndi.affine_transform(
                img,
                inv_rotation,
                order=2,
                offset=offset,
                output=np.int32,
                mode="nearest"
            ))
            f_filename = filename.split('.jpg')[0]
            output_filename = f_filename+'_shear_'+str(i)+'.jpg'

            path = r'./result/shear/%s%s' % (path_dir[11:], output_filename)
            print(path)
            cv2.imwrite(path, out)


print("\n * Completed!!!!")

#cv2.imshow('Original', img)
#cv2.imshow('warpAffine', dst)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
