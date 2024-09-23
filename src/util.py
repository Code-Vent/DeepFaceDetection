import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans


def render_bbox(img, coords, width, height, tag=''):
    cv.rectangle(img, 
                 tuple(np.multiply(coords[:2], [width, height]).astype(int)),
                 tuple(np.multiply(coords[2:], [width, height]).astype(int)),
                 (255, 0, 0),
                 2
                 )
    cv.rectangle(img, 
                 tuple(np.add(np.multiply(coords[:2], [width, height]).astype(int), [0, -30])),
                 tuple(np.add(np.multiply(coords[:2], [width, height]).astype(int), [80, 0])),
                 (255, 0, 0),
                 -1
                 )
    cv.putText(img, tag,
                 tuple(np.add(np.multiply(coords[:2], [width, height]).astype(int), [0, -5])),
                 cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2,cv.LINE_AA)
    return img