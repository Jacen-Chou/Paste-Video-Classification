# -_- coding: utf-8 -_

import os

density = 200
while density <= 780:
    path = './images_ash_sand_1_16_gamma_test/' + str(density)
    files = os.listdir(path)
    i = 1
    for file in files:
        source_file = os.path.join(path, file)
        os.rename(source_file, os.path.join(path, '%d_%s.jpg' % (density, os.path.basename(file)[-7:-4])))
        i += 1
    if density == 780:
        break
    if density < 600:
        density += 50
    else:
        density += 5