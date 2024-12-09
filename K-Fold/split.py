import shutil
import os

# 定义源文件夹和目标文件夹路径
src_images_folder = "D:/develop/datasets/UA-DETRAC_K-Fold/images"  # 源文件夹路径images
src_labels_folder = "D:/develop/datasets/UA-DETRAC_K-Fold/labels"  # 源文件夹路径labels

dst_folder = "D:/develop/datasets/UA-DETRAC_K-Fold/split"  # 目标文件夹路径 /split1/labels/train

# 检查目标文件夹是否存在，如果不存在则创建
if not os.path.exists(dst_folder):
    print("文件不存在")
k_fold = 5
for i in range(k_fold):
    n = i * 2174 + 1
    m = n + 2174 - 1
    current_index = 1
    os.makedirs(dst_folder + f'/split{i}/images/val')
    os.makedirs(dst_folder + f'/split{i}/images/train')
    for filename in os.listdir(src_images_folder):
        src_images_file = os.path.join(src_images_folder, filename)
        dst_val_images_file = os.path.join(dst_folder + f'/split{i}/images/val', filename)
        dst_train_images_file = os.path.join(dst_folder + f'/split{i}/images/train', filename)
        # 确保只复制文件，跳过文件夹
        if os.path.isfile(src_images_file):
            # 检查文件是否在指定的索引范围内
            if n <= current_index <= m:
                shutil.copy(src_images_file, dst_val_images_file)
                print(f"文件 {filename} 已复制到 {dst_folder}/split{i}/images/val")
            else:
                shutil.copy(src_images_file, dst_train_images_file)
                print(f"文件 {filename} 已复制到 {dst_folder}/split{i}/images/train")
            # 更新当前文件索引
            current_index += 1
        # 如果复制到第 m 个文件，停止复制

    current_index = 1
    os.makedirs(dst_folder + f'/split{i}/labels/val')
    os.makedirs(dst_folder + f'/split{i}/labels/train')
    for filename in os.listdir(src_labels_folder):
        src_labels_file = os.path.join(src_labels_folder, filename)
        dst_val_labels_file = os.path.join(dst_folder + f'/split{i}/labels/val', filename)
        dst_train_labels_file = os.path.join(dst_folder + f'/split{i}/labels/train', filename)
        # 确保只复制文件，跳过文件夹
        if os.path.isfile(src_labels_file):
            # 检查文件是否在指定的索引范围内
            if n <= current_index <= m:
                shutil.copy(src_labels_file, dst_val_labels_file)
                print(f"文件 {filename} 已复制到 {dst_folder}/split{i}/labels/val")
            else:
                shutil.copy(src_labels_file, dst_train_labels_file)
                print(f"文件 {filename} 已复制到 {dst_folder}/split{i}/labels/train")
            # 更新当前文件索引
            current_index += 1
        # 如果复制到第 m 个文件，停止复制


