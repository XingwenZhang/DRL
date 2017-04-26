#!/bin/python
# split train and development data
import numpy
import os
os.environ['GLOG_minloglevel'] = '2'
import sys
import caffe
import cPickle

def extract_image_feature(imgs, net, img_transformer):
    features = []
    for i in xrange(len(features)):
        net.blobs['data'].data[...] = img_transformer.preprocess('data', imgs[i, :, :, :])
        net.forward(end = 'pool5')
        features.append(net.blobs['pool5'].data.mean(0).mean(1).mean(1))
    features = numpy.array(features)
    return features

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} input_images_npy output_npy".format(sys.argv[0])
        print "input_images_npy -- path to the input image npy file"
        print "output_npy -- path to the output feature npy file"
        exit(1)
    
    # load caffe models
    resnet_root = '../../deep-residual-networks'
    caffe.set_mode_gpu()
    model_def = resnet_root + 'prototxt/ResNet-152-deploy.prototxt'
    model_weights = resnet_root + 'pretrain_models/ResNet-152-model.caffemodel'

    # initialzie models
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # set batch size
    net.blobs['data'].reshape(1,     # batch size
                              3,         # 3-channel (BGR) images
                              224, 224)  # image size is 224x224

    # load the mean image (as distributed with Caffe) for subtraction
    blob = caffe.proto.caffe_pb2.BlobProto()
    mu = open( resnet_root + 'resnet_root/ResNet_mean.binaryproto' , 'rb' ).read()
    blob.ParseFromString(mu)
    mu = numpy.array( caffe.io.blobproto_to_array(blob) )
    mu = mu.mean(0).mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    # load image here
    imgs = numpy.load(sys.argv[1])
    features = extract_image_feature(imgs, net, transformer)
    numpy.save(sys.argv[2], features)
