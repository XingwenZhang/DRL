#!/bin/python
# split train and development data
import numpy
import os
os.environ['GLOG_minloglevel'] = '2'
import sys
import caffe
import cPickle
import THORUtils as utils
import THORConfig as config

from THOROfflineEnv import ImageDB
from THOROfflineEnv import FeatureDB

feature_layer_name = 'pool5'

def extract_image_feature(img_db, net, img_transformer):
    feat_db = FeatureDB()
    for i in xrange(img_db.get_size()):
        print('extracting feature from image {0}/{1}'.format(i, img_db.get_size()))
        img = img_db.get_img(i)
        net.blobs['data'].data[...] = img_transformer.preprocess('data', img)
        net.forward(end = feature_layer_name)
        feat = net.blobs[feature_layer_name].data.mean(0).mean(1).mean(1)
        feat_db.register_feat(feat)
    feat_db.optimize_memory_layout()
    return feat_db

if __name__ == '__main__':

    # load caffe models
    resnet_root = '../deep-residual-networks'
    caffe.set_mode_gpu()
    model_def = resnet_root + '/pretrain_models/ResNet-50-deploy.prototxt'
    model_weights = resnet_root + '/pretrain_models/ResNet-50-model.caffemodel'

    # initialzie models
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # set batch size
    net.blobs['data'].reshape(1, # batch size
                              3,          # 3-channel (BGR) images
                              224, 224)   # image size is 224x224

    # load the mean image (as distributed with Caffe) for subtraction
    blob = caffe.proto.caffe_pb2.BlobProto()
    mu = open( resnet_root + '/pretrain_models/ResNet_mean.binaryproto' , 'rb' ).read()
    blob.ParseFromString(mu)
    mu = numpy.array( caffe.io.blobproto_to_array(blob) )
    mu = mu.mean(0).mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    # extract features
    if not os.path.exists(config.env_feat_folder):
        os.makedirs(config.env_feat_folder)
    for env in config.supported_envs:
        env_path = "%s/%s.env" %(config.env_db_folder, env)
        feat_path = "%s/%s.feat" %(config.env_feat_folder, env)
        #print env_path
        #print feat_path
        if os.path.exists(env_path) and not os.path.exists(feat_path):
            print('loading image db (%s), this might take a while...' %(env))
            blob = utils.load(open(env_path, 'rb'))
            img_db, mapping = blob
            feat_db = extract_image_feature(img_db, net, transformer)
            blob = (feat_db, mapping)
            utils.dump(blob, open(feat_path, 'wb'))
