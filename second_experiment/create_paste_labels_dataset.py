"""
Create a paste labels dataset.
"""
import csv

dirs_1 = ['train', 'validation', 'test']
dirs_2 = []
density = 200
while density <= 780:
    dirs_2.append(str(density))
    if density == 780:
        break
    if density < 600:
        density += 50
    else:
        density += 5
print(dirs_2)
with open('ash_sand_1_16_paste_labels.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    header = ['image_name_train', 'label', 'image_name_validation', 'label', 'image_name_test', 'label']
    csv_writer.writerow(header)

    for k, lay2 in enumerate(dirs_2):
        for i in range(1, 1001):
            image_name1 = lay2 + '/train_' + lay2 + '_' + str(i) + '.jpg'
            image_name2 = lay2 + '/val_' + lay2 + '_' + str(i) + '.jpg'
            image_name3 = lay2 + '/test_' + lay2 + '_' + str(i) + '.jpg'
            row = [image_name1, k, image_name2, k, image_name3, k]
            csv_writer.writerow(row)
    for k, lay2 in enumerate(dirs_2):
        for i in range(1001, 3001):
            image_name = lay2 + '/train_' + lay2 + '_' + str(i) + '.jpg'
            row = [image_name, k]
            csv_writer.writerow(row)
