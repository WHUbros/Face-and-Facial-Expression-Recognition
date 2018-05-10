import os
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
import matplotlib.pyplot as plt

mean_image = np.load('C:/Users/zekun/Desktop/Bigdata/project/mean_image.npy')
dic = {0:'surprised',1:'sad',2:'angry',3:'happy',4:'neutral'}
IMG_SIZE = 48

class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, index=0):
        #         print("len(input_x.shape),input_x.shape[1],input_x.shape[2],input_x.shape[3],in_channel",len(input_x.shape),input_x.shape[1],input_x.shape[2],input_x.shape[3],in_channel)
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.contrib.layers.xavier_initializer())
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.contrib.layers.xavier_initializer())
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, 1, 1, 1], padding="SAME")
            cell_out = tf.nn.relu(conv_out + bias)

            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class norm_layer(object):
    def __init__(self, input_x):
        with tf.variable_scope('batch_norm'):
            mean, variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=None,
                                                 scale=None,
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed, activation_function=tf.nn.relu, index=0):
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.contrib.layers.xavier_initializer())
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.contrib.layers.xavier_initializer())
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            self.cell_out = cell_out

    def output(self):
        return self.cell_out


def my_Net(input_x,
           img_len=IMG_SIZE, channel_num=1, output_size=5,
           conv_featmap=[32, 64, 128], fc_units=[84],
           conv_kernel_size=[3, 3, 3], pooling_size=[2, 2, 2], seed=235):
    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)


    conv_layer_0_1 = conv_layer(input_x=input_x,
                                in_channel=channel_num,
                                out_channel=conv_featmap[0],
                                kernel_shape=conv_kernel_size[0],
                                rand_seed=seed, index=0)

    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_0_1.output(),
                                        k_size=pooling_size[0],
                                        padding="VALID")

    pool_shape = pooling_layer_0.output().get_shape()

    norm_layer_0 = norm_layer(pooling_layer_0.output())

    ######### conv layer *3
    conv_layer_1_1 = conv_layer(input_x=norm_layer_0.output(),
                                in_channel=pool_shape[3],
                                out_channel=conv_featmap[1],
                                kernel_shape=conv_kernel_size[0],
                                rand_seed=seed, index=1)

    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_1_1.output(),
                                        k_size=pooling_size[1],
                                        padding="VALID")

    pool1_shape = pooling_layer_1.output().get_shape()

    norm_layer_1 = norm_layer(pooling_layer_1.output())

    #######################################

    conv_layer_2_1 = conv_layer(input_x=norm_layer_1.output(),
                                in_channel=pool1_shape[3],
                                out_channel=conv_featmap[2],
                                kernel_shape=conv_kernel_size[0],
                                rand_seed=seed, index=2)

    pooling_layer_2 = max_pooling_layer(input_x=conv_layer_2_1.output(),
                                        k_size=pooling_size[1],
                                        padding="VALID")

    norm_layer_2 = norm_layer(pooling_layer_2.output())

    # flatten
    pool_shape1 = norm_layer_2.output().get_shape()
    img_vector_length = pool_shape1[1].value * pool_shape1[2].value * pool_shape1[3].value
    flatten = tf.reshape(norm_layer_2.output(), shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=None,
                          index=0)

    fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                          in_size=fc_units[0],
                          out_size=output_size,
                          rand_seed=seed,
                          activation_function=None,
                          index=1)

    return fc_layer_1.output()


def predict(output):
    with tf.name_scope('predict'):
        pred = tf.argmax(output, axis=1)
    return pred

def rgb2gray(rgb):
    a = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return a
def get_images():
    count = 1
    mytest = []
    # mytest.append(np.zeros([]))
    mytest.append(np.zeros([IMG_SIZE,IMG_SIZE,1]))
    for i in range(1,5):
        count += 1
        img = np.array(Image.open('C:/Users/zekun/Desktop/Bigbata/Myimages/'+ str(i) + '.jpg'))
        print(img.shape)
        gray_img = rgb2gray(img)
        img = imresize(gray_img,(IMG_SIZE,IMG_SIZE))
        img = np.reshape((img.astype(np.float32) - mean_image.astype(np.float32))/255,[IMG_SIZE,IMG_SIZE,1])
        print(img.shape, gray_img.shape)
        mytest.append(img)
    mytest = np.array(mytest)
    mytest = np.reshape(mytest,(count, IMG_SIZE, IMG_SIZE, 1))
    return mytest

#### Prediction

tf.reset_default_graph()
with tf.name_scope('inputs'):
    xs = tf.placeholder(shape=[None, IMG_SIZE, IMG_SIZE, 1], dtype=tf.float32)
output = my_Net(xs)
pre = predict(output)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, 'C:/Users/zekun/Desktop/Bigdata/project/model/lenet_1522818805')
    counter = 0
    while counter < 1:
        x_test = get_images()
        length = x_test.shape[0]
        prediction = sess.run([pre], feed_dict={xs: x_test})[0]
        for i in prediction:
            print(dic[i])
        counter += 1