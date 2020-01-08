import os
import cv2
import numpy as np

density = 200
while density <= 780:
    path = './images_ash_sand_1_16_gamma_test/' + str(density)
    files = os.listdir(path)
    for file in files:
        print(os.path.basename(file))
        source_file = os.path.join(path, file)
        img = cv2.imread(source_file, 0)
        gamma = 0.5
        while gamma < 1.5:
            img_gamma = np.power(img / float(np.max(img)), gamma) * 255
            cv2.imwrite(os.path.join(path, os.path.basename(file)[0:-4] + '_%.1f.jpg' % gamma), img_gamma)
            gamma += 0.1
        os.remove(os.path.join(path, os.path.basename(file)[0:-4] + '_1.0.jpg'))
        os.rename(source_file, os.path.join(path, os.path.basename(file)[0:-4] + '_1.0.jpg'))
    if density == 780:
        break
    if density < 600:
        density += 50
    else:
        density += 5


