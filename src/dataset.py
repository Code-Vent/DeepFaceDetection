import tensorflow as tf
import json
import numpy as np
import os
import preprocess

TRAIN_ROOT = '../aug_data/train'
TEST_ROOT = '../aug_data/test'
VAL_ROOT = '../aug_data/val'
    
    
def __load_image(filename):
    img = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img)
    return img

def __load_label(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        label = json.load(f)
    return [label['class']], label['bboxes']

def __create_imageset(root, shuffle=False):
    images = tf.data.Dataset.list_files(os.path.join(root, 'images', '*.jpg'), shuffle)
    #print(np.shape(images))
    images = images.map(__load_image)
    images = images.map(lambda x: tf.image.resize(x, (120,120)))
    images = images.map(lambda x: x/255)
    return images

def __create_labelset(root, shuffle=False):
    labels = tf.data.Dataset.list_files(os.path.join(root, 'labels', '*.json'), shuffle)
    classes = []
    boxes   = []
    for e in labels.as_numpy_iterator():
        a, b = __load_label(e)
        classes.append(a)
        boxes.append(b)
    classes = tf.data.Dataset.from_tensor_slices(np.array(classes, dtype=np.int8))
    boxes   = tf.data.Dataset.from_tensor_slices(np.array(boxes, dtype=np.float16))
    return tf.data.Dataset.zip((classes, boxes))
    
def __create_data_batches(imageset, labelset, buffer_size):
    data = tf.data.Dataset.zip((imageset, labelset))
    data = data.shuffle(buffer_size).batch(8).prefetch(4)
    return data
    
#preprocess.run_augmentation()

def get():
    train_images = __create_imageset(TRAIN_ROOT)
    test_images  = __create_imageset(TEST_ROOT)
    val_images   = __create_imageset(VAL_ROOT)

    train_labels = __create_labelset(TRAIN_ROOT)
    test_labels  = __create_labelset(TEST_ROOT)
    val_labels   = __create_labelset(VAL_ROOT)

    train = __create_data_batches(train_images, train_labels, 5000)
    test = __create_data_batches(test_images, test_labels, 1500)
    val = __create_data_batches(val_images, val_labels, 1500)
    return train, test, val
