import skimage.io  
import skimage.transform  
import tensorflow as tf
import numpy as np
import os


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
def fc(x, num_units_out):
    
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

  #Pretrained resnet to take in the 4 observation frames
  with tf.variable_scope("fc1"):
    #Take the 2048d feature from pretrained ResNet
    output_conv1 =graph.get_tensor_by_name('avg_pool:0')
    #Tensor to save input image 
    images1 = graph.get_tensor_by_name("images:0")
    #This tensor is supposed freeze and store the ResNet parameters, not sure...
    output_conv_sg1 = tf.stop_gradient(output_conv1) 
    # 1x2048 feature vector -->fc layer --> 1x512
    fc1_out = fc(output_conv_sg1, 512)

  #Pretrained resenet to take in the target location image  
  with tf.variable_scope("fc2"):
    output_conv2=graph.get_tensor_by_name('avg_pool:0')
    images2 = graph.get_tensor_by_name("images:0")
    output_conv_sg2 = tf.stop_gradient(output_conv2) 
    fc2_out = fc(output_conv_sg2, 512)

  #concatenate the two outputs from fc1 and fc2, pass through fc3
  with tf.variable_scope("fc3"):
    fc3_in = tf.concat([fc1_out , fc2_out],1)
    fc3_out = fc(fc3_in,512)
  #policy(4)
  with tf.variable_scope("fc4_policy"):
    fc4_policy_out = fc(fc3_out,4)
  #value(1)
  with tf.variable_scope("fc5_value"):
    fc5_value_out = fc(fc3_out,1)
  


  #TODO: add action and vallue, end to end, return action&value
  train_writer = tf.summary.FileWriter(train_dir,
                                      sess.graph)
#TODO: question-- how to define the input image for the graph? how to connect fc3 to action?
run_graph()





