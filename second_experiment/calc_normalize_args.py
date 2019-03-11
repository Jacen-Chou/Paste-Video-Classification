# 用训练集样本计算归一化参数

import os
import numpy as np
import cv2
import time
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

temp_tensor = torch.zeros(1080, 1920, 3, dtype=torch.float)
temp_tensor = temp_tensor.cuda()
density = 200  # 初始浓度

f = open('./result/calc_normalize_args.txt', 'w')
since = time.time()

while True:
    # 把所有图片的RGB值分别加起来，存入temp
    for num_train in range(1, 3001):
        img_path = './images_paste/images_ash_sand_1_16/train/' + str(density) + '/train_' + str(density) \
               + '_' + str(num_train) + '.jpg'
        img = cv2.imread(img_path)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np)
        img_tensor = img_tensor.cuda()
        new_img_tensor = img_tensor.float()
        new_img_tensor = new_img_tensor.cuda()
        temp_tensor = torch.add(temp_tensor, new_img_tensor)
    time_elapsed = time.time() - since
    print(density, end = ' ')
    print('Time elapsed {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
    f.write('%d' % density)
    f.write('Time elapsed {:.0f}h {:.0f}m {:.0f}s\n'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

    if density == 780:
        break
    if density < 600:
        density += 50
    else:
        density += 5

temp_np = temp_tensor.numpy()
# 对temp切片，分别取出R、G、B的和
R = np.copy(temp_np[..., 0])
G = np.copy(temp_np[..., 1])
B = np.copy(temp_np[..., 2])

# 求平均值
R_mean = R.mean()/3000/255
G_mean = G.mean()/3000/255
B_mean = B.mean()/3000/255

# 求标准差
R_std = R.std()/3000/255
G_std = G.std()/3000/255
B_std = B.std()/3000/255

print('R_mean: %d' % R_mean)
print('G_mean: %d' % G_mean)
print('B_mean: %d' % B_mean)
print('R_std: %d' % R_std)
print('G_std: %d' % G_std)
print('B_std: %d' % B_std)

f.write('R_mean: %d\n' % R_mean)
f.write('G_mean: %d\n' % G_mean)
f.write('B_mean: %d\n' % B_mean)
f.write('R_std: %d\n' % R_std)
f.write('G_std: %d\n' % G_std)
f.write('B_std: %d\n' % B_std)
f.close()