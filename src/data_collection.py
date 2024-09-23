import os
import time
import uuid
import cv2 as cv

IMAGE_PATH = os.path.join('../data' , 'images')
number_images = 30


def capture_images(img_path: str, num_images: int)->None:
   cap = cv.VideoCapture(1) 
   for i in range(num_images):
       print('collecting image {}'.format(i))
       ret, img = cap.read()
       filename = os.path.join(img_path, f'{uuid.uuid1()}.jpg')
       cv.imwrite(filename, img)
       cv.imshow('frame', img)
       time.sleep(0.5)
       
       if cv.waitKey(1) & 0xFF == ord('q'):
           break
       
   cap.release()
   cv.destroyAllWindows()
   
   
   
if __name__ == '__main__':
    capture_images(IMAGE_PATH, number_images)


