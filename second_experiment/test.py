import os

# 删除一个目录下所有文件和子文件夹中的所有文件
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

# 如果之前有图片遗留，则清空

path = "./tttt/"
os.rmdir(path)