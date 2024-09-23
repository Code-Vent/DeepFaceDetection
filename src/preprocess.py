import os




for folder in ['train', 'val', 'test']:
    for file in os.listdir(os.path.join('../data', folder, 'images')):
        filename = file.split('.')[0] + '.json'
        old_file_path = os.path.join('../data', 'labels', filename)
        if os.path.exists(old_file_path):
            new_file_path = os.path.join('../data', folder, 'labels', filename)
            os.replace(old_file_path, new_file_path)
            

import albumentations as alb
import cv2 as cv
import json
import numpy as np
            
augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),alb.VerticalFlip(p=0.5)],
    bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

def run_augmentation():
    print('running augmentation#########')
    for partition in ['train', 'test', 'val']:
        root = os.path.join('../data', partition)
        aug_root = os.path.join('../aug_data', partition)
        image_path = os.path.join(root, 'images')
        for image in os.listdir(image_path):
            img = cv.imread(os.path.join(image_path, image))
            coords = [0,0,0.00001,0.00001]
            label_path = os.path.join(root, 'labels', f'{image.split('.')[0]}.json')
            if os.path.exists(label_path):
                with open(label_path) as f:
                    label = json.load(f)
                coords[0] = label['shapes'][0]['points'][0][0]
                coords[1] = label['shapes'][0]['points'][0][1]
                coords[2] = label['shapes'][0]['points'][1][0]
                coords[3] = label['shapes'][0]['points'][1][1]
                
                coords = list(np.divide(coords , [640, 480, 640, 480]))
            
            try:
                for x in range(50):
                    augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                    aug_image_path = os.path.join(aug_root, 'images', f'{image.split('.')[0]}.{x}.jpg')
                    cv.imwrite(aug_image_path, augmented['image'])
                    
                    annotation = {}
                    annotation['image'] = image
                    annotation['bboxes'] = [0,0,0,0]
                    annotation['class']  = 0
                    if os.path.exists(label_path) and len(augmented['bboxes']) > 0:
                        annotation['bboxes'] = augmented['bboxes'][0]
                        annotation['class']  = 1

                    annotation_path = os.path.join(aug_root, 'labels', f'{image.split('.')[0]}.{x}.json')
                    with open(annotation_path, 'w') as f:
                        json.dump(annotation, f)
                        
            except Exception as e:
                print(e)