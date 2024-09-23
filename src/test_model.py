import tensorflow as tf
import dataset
import util
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

_, test, _ = dataset.get()


facetracker = load_model('facetracker.h5')
it = test.as_numpy_iterator()

while True:
    test_images, _ = it.next()
    classes, coords = facetracker(test_images, training=False)


    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx in range(4):
        img = test_images[idx].copy()
        img = util.render_bbox(img, coords[idx], 120, 120, 'face')
        ax[idx].imshow(img)
    
    plt.show()
    ans = input('continue (y/n)? ')
    if ans.lower() == 'n':
        exit()
