import os
import cv2
import numpy as np
import argparse

def load_images_from_folders(folders):
    images_per_folder = []
    for folder in folders:
        images = {}
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images[filename] = img
        images_per_folder.append(images)
    return images_per_folder

def blend_images(images_per_folder, weights):
    # 对图像进行加权融合
    blended_images = {}
    for images, weight in zip(images_per_folder, weights):
        for filename, img in images.items():
            if filename in blended_images:
                blended_images[filename] += img * weight
            else:
                blended_images[filename] = img * weight
    for filename, img in blended_images.items():
        blended_images[filename] /= np.sum(weights)
        blended_images[filename] = blended_images[filename].astype(np.uint8)
    return blended_images

def main(args):
    # folders = [
    #     "folder1", 
    #     "folder2", 
    #     "folder3",
    #     ]  # 替换为你的文件夹列表
    folders = args.img_folders
    weights = args.weights #[0.3, 0.5, 0.2]  # 替换为每个文件夹对应的权重

    # 加载图像
    images_per_folder = load_images_from_folders(folders)

    # 对每个图像进行加权融合
    blended_images = blend_images(images_per_folder, weights)

    # 保存融合后的图像
    os.makedirs(args.save_path, exist_ok=True)
    for filename, img in blended_images.items():
        cv2.imwrite(os.path.join(args.save_path, filename), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_folders', type=str, nargs='*', default=[''])
    parser.add_argument('--weights', type=float, nargs='*', default=[1.])
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()
    main(args)
