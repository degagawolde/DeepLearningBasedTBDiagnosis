import os
import cv2 
import numpy as np
import tensorflow as tf

def create_dataset(img_folder, IMG_SIZE):
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        i = 0
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path)
            if image is not None:
                image=cv2.resize(image, IMG_SIZE,interpolation = cv2.INTER_CUBIC)
                image=np.array(image)
                image = image.astype('float32')
#                 image /= 255 
                img_data_array.append(image)
                class_name.append(dir1)
            i += 1
            if i%500 == 0:
                print('loading {}th image for class {}'.format(i,str(dir1)))
    return np.array(img_data_array,np.float32), class_name


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array
