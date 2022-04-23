import tensorflow as tf
import sys
import os
import time
import datetime
from utils import *
from pydnet import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

checkpoint_dir = "checkpoint/IROS18/pydnet"
width = 512
height = 256
resolution = 1 #1-H, 2-Q, 3-E
output = "depth1.jpg"
output_img = "image1.jpg"
        
def main(input_image):

  with tf.Graph().as_default():
    placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 4], name='im0')}
    with tf.variable_scope("model") as scope:
      model = pydnet(placeholders)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    loader = tf.train.Saver()

    with tf.Session() as sess:
      sess.run(init)
      loader.restore(sess, checkpoint_dir)
      input_image = cv2.resize(input_image, (width, height))
      img = input_image.astype(np.float32) / 255.
      img = np.expand_dims(img, 0)
      start = time.time()
      disp = sess.run(model.results[resolution-1], feed_dict={placeholders['im0']: img})
      end = time.time()

      disp_color = applyColorMap(disp[0,:,:,0]*20, 'binary')
      depth = (disp_color*255.).astype(np.uint8)
      img = (img*255.).astype(np.uint8)
      depth = cv2.resize(depth, (int(width), int(height)))
      #cv2.imwrite(output, depth)
      #cv2.imwrite(output_img, input_image)
      cv2.imshow(depth)

      print("Time: " + str(end - start)) 

      return depth   

if __name__ == '__main__':
    tf.app.run()
