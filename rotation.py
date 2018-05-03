
import os, sys
import cv2


all_data_dir = ['./data/test/ruptured/base/', './data/test/ruptured/mark/',
                './data/test/unruptured/base/', './data/test/unruptured/mark/',
                './data/train/ruptured/base/', './data/train/ruptured/mark/',
                './data/train/unruptured/base/', './data/train/unruptured/mark/']


for path_dir in all_data_dir:
    file_list = os.listdir(path_dir)

    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')

    for filename in file_list:
        input_filename = path_dir+filename
        print(input_filename)

        img = cv2.imread(input_filename, 1)
        rows, cols = img.shape[:2]

        for i in range(0, 4):
            angle = i*90
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
            dst = cv2.warpAffine(img, M, (cols, rows))

            f_filename = filename.split('.jpg')[0]
            output_filename = f_filename+'_rotate_'+str(angle)+'.jpg'

            path = r'./result/%s%s' % (path_dir[7:], output_filename)
            cv2.imwrite(path, dst)


print("\n * Completed!!!!")

#cv2.imshow('Original', img)
#cv2.imshow('warpAffine', dst)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
