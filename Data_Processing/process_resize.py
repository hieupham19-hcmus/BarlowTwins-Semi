import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def process_polygen(origin_folder, processed_folder):
    image_dir_path = origin_folder + '/images/'
    mask_dir_path = origin_folder + '/masks/'
    
    image_path_list = os.listdir(image_dir_path)
    mask_path_list = os.listdir(mask_dir_path)
    
    image_path_list = list(filter(lambda x: x[-3:] == 'jpg', image_path_list))
    mask_path_list = list(filter(lambda x: x[-3:] == 'jpg', mask_path_list))
    
    image_path_list.sort()
    mask_path_list.sort()
    
    print('number of images: {}, number of masks: {}'.format(len(image_path_list), len(mask_path_list)))
    
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        if image_path[-3:] == 'jpg':
            print(image_path)
            assert os.path.basename(image_path)[:-4].split('_')[1] == os.path.basename(mask_path)[:-4].split('_')[1]
            _id = os.path.basename(image_path)[:-4].split('_')[1]
            image_path = os.path.join(image_dir_path, image_path)
            mask_path = os.path.join(mask_dir_path, mask_path)
            image = plt.imread(image_path)
            mask = plt.imread(mask_path)
            if len(mask.shape) == 3:
                mask = np.int64(np.all(mask[:, :, :3] == 1, axis=2))
            
            image_new = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
            mask_new = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            
            save_dir_path = processed_folder + '/images'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), image_new)
            
            save_dir_path = processed_folder + '/labels'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), mask_new)

def delete_file_mask_bbox(path):
    #"D:\RESEARCH\BT_Semi\data\polypgen\images\EndoCV2021_C6_0100107_mask_bbox.jpg"
    for root, dirs, files in os.walk(path):
        for file in files:
            if file[-3:] == 'jpg':
                if 'mask_bbox' in file:
                    os.remove(os.path.join(root, file))
                    print('deleted:', os.path.join(root, file))
                    
      
if __name__ == '__main__':
    delete_file_mask_bbox('D:/RESEARCH/BT_Semi/data/polypgen/images')