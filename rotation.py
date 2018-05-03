
import os, sys
import cv2


all_data_dir = ['./data/test/ruptured/base/', './data/test/ruptured/mark/',
                './data/test/unruptured/base/', './data/test/unruptured/mark/',
                './data/train/ruptured/base/', './data/train/ruptured/mark/',
                './data/train/unruptured/base/', './data/train/unruptured/mark/']

for path_dir in all_data_dir:
    #path_dir = './data/test/ruptured/base/'
    file_list = os.listdir(path_dir)

    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')

    for filename in file_list:
        input_filename = path_dir+filename
        print(input_filename)

        for i in range(1, 4):
            img = cv2.imread(input_filename, 1)
            rows, cols = img.shape[:2]
            angle = i*90

            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
            dst = cv2.warpAffine(img, M, (cols, rows))

            f_filename = filename.split('.jpg')[0]

            output_filename = f_filename+'_rotate_'+str(angle)+'.jpg'
            cv2.imwrite('./result/'+path_dir[7:]+output_filename, dst)


print("\n * Completed!!!!");

#cv2.imshow('Original', img)
#cv2.imshow('warpAffine', dst)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
