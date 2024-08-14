import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import random
from sklearn.model_selection import KFold

def extract_mask_id(mask_name):
    """
    Extracts the ID from a mask filename using regex.
    """
    match = re.match(r'(.+?)_mask\.jpg$', mask_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Filename {mask_name} does not match expected pattern.")

def extract_number(filename):
    """
    Extracts the leading number from a filename for numerical sorting.
    """
    match = re.match(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def process_polypgen(origin_folder, processed_folder):
    """
    Processes images and corresponding masks by resizing them to 512x512, and saving them as .npy files.
    """
    image_dir_path = os.path.join(origin_folder, 'images')
    mask_dir_path = os.path.join(origin_folder, 'masks')

    # Get lists of image and mask files with numerical sorting
    image_files = sorted([f for f in os.listdir(image_dir_path) if f.endswith('.jpg')], key=extract_number)
    mask_files = sorted([f for f in os.listdir(mask_dir_path) if f.endswith('_mask.jpg')], key=extract_number)

    print(f'Number of images: {len(image_files)}, Number of masks: {len(mask_files)}')
    
    # Processing images and masks
    for image_name, mask_name in zip(image_files, mask_files):
        image_id = os.path.splitext(image_name)[0]
        mask_id = extract_mask_id(mask_name)

        if image_id != mask_id:
            print(f"ID mismatch: {image_name} and {mask_name}")
            continue

        # Read, resize, and save the image and mask
        image_path = os.path.join(image_dir_path, image_name)
        mask_path = os.path.join(mask_dir_path, mask_name)

        image = plt.imread(image_path)
        mask = plt.imread(mask_path)
        
        # Ensure the mask is binary (single channel)
        if mask.ndim == 3:
            # Assuming a colored mask where white (255, 255, 255) represents the mask
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        elif mask.ndim == 2 and np.unique(mask).size > 2:
            # If the mask is grayscale with more than 2 values, threshold it
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Save resized images and masks
        np.save(os.path.join(processed_folder, 'images', f'{image_id}.npy'), image_resized)
        np.save(os.path.join(processed_folder, 'labels', f'{mask_id}.npy'), mask_resized)

    print("Processing completed.")

def numeric_sort_key(filename):
    # Extract the numeric part of the filename (before the '.npy')
    return int(os.path.splitext(filename)[0])

def split_into_folds(processed_folder, num_folds=5, seed=42):
    """
    Splits processed data into folds, sorts the filenames numerically within each fold,
    and saves the sorted file names of each fold into .txt files.

    Parameters:
    - processed_folder: The folder where processed images and labels are stored.
    - num_folds: The number of folds to split the data into.
    - seed: The seed for random number generation to ensure reproducibility.
    """
    image_dir = os.path.join(processed_folder, 'images')

    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])

    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create KFold object
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for fold, (_, val_idx) in enumerate(kf.split(image_files), 1):
        fold_file = os.path.join(processed_folder, f'fold{fold}.txt')

        # Extract filenames for this fold
        fold_filenames = [image_files[idx] for idx in val_idx]

        # Sort the filenames numerically before saving
        fold_filenames.sort(key=numeric_sort_key)

        # Save the sorted validation file names into foldX.txt
        with open(fold_file, 'w') as fold_txt:
            for filename in fold_filenames:
                fold_txt.write(f'{filename}\n')
    
    print(f"Data split into {num_folds} numerically sorted folds and saved to fold1.txt, fold2.txt, ... fold{num_folds}.txt.")


def clean_and_rename(base_dir):
    """
    This function removes any unmatched images or masks in the specified directory
    and renames the remaining image-mask pairs to have sequential filenames.
    
    Parameters:
    base_dir (str): Path to the base directory containing 'images' and 'masks' folders.
    """
    
    image_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    
    # List all image and mask files
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Convert lists to sets for easier comparison
    image_basenames = {os.path.splitext(f)[0] for f in image_files}
    mask_basenames = {os.path.splitext(f)[0].replace('_mask', '') for f in mask_files}

    # Find unmatched files
    unmatched_images = image_basenames - mask_basenames
    unmatched_masks = mask_basenames - image_basenames

    # Remove unmatched images
    for file in image_files:
        basename = os.path.splitext(file)[0]
        if basename in unmatched_images:
            os.remove(os.path.join(image_dir, file))

    # Remove unmatched masks
    for file in mask_files:
        basename = os.path.splitext(file)[0].replace('_mask', '')
        if basename in unmatched_masks:
            os.remove(os.path.join(mask_dir, file))

    # Refresh the list of files after removal
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Rename images and masks
    for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        # Rename the image
        new_image_name = f"{i}.jpg"
        image_path = os.path.join(image_dir, image_file)
        new_image_path = os.path.join(image_dir, new_image_name)
        os.rename(image_path, new_image_path)
        
        # Rename the corresponding mask
        new_mask_name = f"{i}_mask.jpg"
        mask_path = os.path.join(mask_dir, mask_file)
        new_mask_path = os.path.join(mask_dir, new_mask_name)
        os.rename(mask_path, new_mask_path)

    print("Cleaning and renaming completed.")
                    
      
#if __name__ == '__main__':
    