
import os, sys
import cv2


path_dir = './data/test/ruptured/base/'
file_list = os.listdir(path_dir)

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
        cv2.imwrite('./result/'+output_filename, dst)


#cv2.imshow('Original', img)
#cv2.imshow('warpAffine', dst)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
