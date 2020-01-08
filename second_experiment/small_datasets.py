# -_- coding: utf-8 -_

import os


# 复制文件
def copy_files(source_dir, target_dir, length):
    i = 0
    files = os.listdir(source_dir)
    files.sort(key=lambda x: int(x[length:-4]))
    for file in files:
        i += 1
        if not i % 5 == 1:
            continue
        else:
            source_file = os.path.join(source_dir, file)
            target_file = os.path.join(target_dir, file)
            if os.path.isfile(source_file):
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                if not os.path.exists(target_file) or (
                        os.path.exists(target_file) and (os.path.getsize(target_file) != os.path.getsize(source_file))):
                    open(target_file, "wb").write(open(source_file, "rb").read())
                os.rename(target_file, os.path.join(target_dir, str(os.path.basename(target_file))[0:length] + str(i / 5 + 1) + '.jpg'))
            if os.path.isdir(source_file):
                copy_files(source_file, target_file)

if __name__ == '__main__':
    dirs = ['validation', 'test']
    for img_dir in dirs:
        density = 200
        while density <= 780:
            source_dir = './images_paste/images_ash_sand_1_16/' + str(img_dir) + '/' + str(density)
            target_dir = './images_paste/images_ash_sand_1_16_little/' + str(img_dir) + '/' + str(density)
            length = 8 if (img_dir == 'validation') else (5 + len(img_dir))
            copy_files(source_dir, target_dir, length)
            print('copy ' + str(img_dir) + ' ' + str(density) + ' done.')
            if density < 600:
                density += 50
            else:
                density += 5



