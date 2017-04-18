#multiple graphs
#https://www.quora.com/How-does-one-create-separate-graphs-in-TensorFlow
#http://stackoverflow.com/questions/35955144/working-with-multiple-graphs-in-tensorflow
import tensorflow as tf
import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

import datetime
import numpy as np
import os
import time



FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'

#directory to save log files for tensorboard 
dir = os.path.dirname(os.path.realpath(__file__)) + 'logs/'

#helper function to load images as 224x224x3 size
def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img

#A little wrapper around tf.get_variable to do weight decay 
def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


#Define the fc layer to take feature vectore from pretrained ResNet
def fc(x, num_units_out=512):
    
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x
#Input 4+1 images, pass through pretrianed ResNet, add fc layers at the end
def run_graph(train_dir='2graph_4img_logs'):

  path_model = './ResNet-L50.meta'
  path_model2 = './ResNet-L50.ckpt'
  
  init = tf.global_variables_initializer()
  
  sess = tf.Session()
  sess.run(init)

  saver = tf.train.import_meta_graph(path_model)
  saver.restore(sess, path_model2)
  graph = tf.get_default_graph()

  #4 images, corresponds to the 4 observation frames
  img1 = load_image("room.jpg")
  img2 = load_image("room.jpg")
  img3 = load_image("room.jpg")
  img4 = load_image("room.jpg")


  #target image
  img5 = load_image("room.jpg")
  img5 = img5.reshape((1,224, 224, 3))
  
  #Pretrained resnet to take in the 4 observation frames
  with tf.variable_scope("fc1"):
    #Take the 2048d feature from pretrained ResNet
    output_conv =graph.get_tensor_by_name('avg_pool:0')
    #Tensor to save input image 
    images = graph.get_tensor_by_name("images:0")
    #This tensor is supposed freeze and store the ResNet parameters, not sure...
    output_conv_sg = tf.stop_gradient(output_conv) 

  #Pretrained resenet to take in the target location image  
  with tf.variable_scope("fc2"):
    output_conv2=graph.get_tensor_by_name('avg_pool:0')
    images2 = graph.get_tensor_by_name("images:0")
    output_conv_sg2 = tf.stop_gradient(output_conv2) 

  #Input 4 observations and the 1 target frame
  output_1 = sess.run([output_conv_sg], feed_dict={images: [img1,img2,img3,img4]})
  output_2 = sess.run([output_conv_sg2], feed_dict={images:img5})
  #Convert list format output_1, output_2 to tensor
  output_1_tf = tf.stack(output_1)
  output_2_tf = tf.stack(output_2)
  #Reshape from 1x1x2048 to 1x2048
  output_1_tf = tf.reshape(output_1_tf,[1,-1])
  output_2_tf = tf.reshape(output_2_tf,[1,-1])
  # 1x2048 feature vector -->fc layer --> 1x512
  with tf.variable_scope("fc1"):
    fc_output = fc(output_1_tf, 512)
  with tf.variable_scope("fc2"):
    fc_output2 = fc(output_2_tf, 512)
  #concatenate the two outputs from fc1 and fc2, pass through fc3
  with tf.variable_scope("fc3"):
    fc_concat = tf.concat([fc_output , fc_output2],1)
    fc_3 = fc(fc_concat,512)

  train_writer = tf.summary.FileWriter(train_dir,
                                      sess.graph)

run_graph()