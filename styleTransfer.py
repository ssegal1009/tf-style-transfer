import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2


model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) #data type is float32
    img = img[tf.newaxis, :]#make sure image is inside of a new array
    return img


#visualize output:
content_image = load_image('profile.jpg')
style_image = load_image('monet.jpg')

#plt.imshow(np.squeeze(style_image))
#plt.show()


#stylize the image:

stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
plt.imshow(np.squeeze(stylized_image))
plt.show()
