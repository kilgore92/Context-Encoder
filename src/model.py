import tensorflow as tf
import numpy as np
import math
import pickle

class Model():
    def __init__(self, image_size, hiding_size, batch_size):

        self.image_size = image_size
        self.hiding_size = hiding_size

        self.batch_size = batch_size

        # each layer will do 128 x 128 -> 64 x 64
        encoderLayerNum = int(math.log(self.image_size) / math.log(2))
        encoderLayerNum = encoderLayerNum - 1 # minus 1 because the second last layer directly go from 4x4 to 1x1
        print("encoderLayerNum = {}".format(encoderLayerNum))
        self.encoderLayerNum = encoderLayerNum

        # Implementation out of sync with the paper??
        # Input to the encoder is the image, why is the decoder output only the patch ??
        decoderLayerNum = int(math.log(self.hiding_size) / math.log(2))
        decoderLayerNum = decoderLayerNum - 1
        print("decoderLayerNum=", decoderLayerNum)
        self.decoderLayerNum = decoderLayerNum

        pass

    def new_conv_layer( self, bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
        with tf.variable_scope( name ):
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias #relu

    def new_deconv_layer(self, bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-2],
                    initializer=tf.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))

        return bias

    def new_fc_layer( self, bottom, output_size, name ):
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])
        input_size = dim

        with tf.variable_scope(name):
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b)

        return fc

    def channel_wise_fc_layer(self, input, name): # bottom: (7x7x512)
        _, width, height, n_feat_map = input.get_shape().as_list()
        input_reshape = tf.reshape( input, [-1, width*height, n_feat_map] )
        input_transpose = tf.transpose( input_reshape, [2,0,1] )

        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=[n_feat_map,width*height, width*height], # (512,49,49)
                    initializer=tf.random_normal_initializer(0., 0.005))
            output = tf.batch_matmul(input_transpose, W)

        output_transpose = tf.transpose(output, [1,2,0])
        output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

        return output_reshape

    def leaky_relu(self, bottom, leak=0.1):
        return tf.maximum(leak*bottom, bottom)

    def batchnorm(self, bottom, is_train, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)
        depth = bottom.get_shape().as_list()[-1]

        # with tf.variable_scope(name):

        #     gamma = tf.get_variable("gamma", [depth], initializer=tf.constant_initializer(1.))
        #     beta  = tf.get_variable("beta" , [depth], initializer=tf.constant_initializer(0.))

        #     batch_mean, batch_var = tf.nn.moments(bottom, [0,1,2], name='moments')
        #     ema = tf.train.ExponentialMovingAverage(decay=0.5)


        #     def update():
        #         with tf.control_dependencies([ema_apply_op]):
        #             return tf.identity(batch_mean), tf.identity(batch_var)

        #     ema_apply_op = ema.apply([batch_mean, batch_var])
        #     ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        #     mean, var = tf.cond(
        #             is_train,
        #             update,
        #             lambda: (ema_mean, ema_var) )

        #     normed = tf.nn.batch_norm_with_global_normalization(bottom, mean, var, beta, gamma, epsilon, False)

        with tf.variable_scope(name):

            normed = tf.contrib.layers.batch_norm(bottom, decay=0.5, epsilon=epsilon, scale=False)


        return normed

    def build_reconstruction( self, images, is_train ):
        """
        Builds the encoder-decoder structure

        """
        batch_size = images.get_shape().as_list()[0]



        encoderLayerNum = self.encoderLayerNum
        decoderLayerNum = self.decoderLayerNum

        with tf.variable_scope('GEN'):
            # conv1 = self.new_conv_layer(images, [4,4,3,64], stride=2, name="conv1" )
            # bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1'))
            # conv2 = self.new_conv_layer(bn1, [4,4,64,64], stride=2, name="conv2" )
            # bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2'))
            # conv3 = self.new_conv_layer(bn2, [4,4,64,128], stride=2, name="conv3")
            # bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3'))
            # conv4 = self.new_conv_layer(bn3, [4,4,128,256], stride=2, name="conv4")
            # bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4'))
            # conv5 = self.new_conv_layer(bn4, [4,4,256,512], stride=2, name="conv5")
            # bn5 = self.leaky_relu(self.batchnorm(conv5, is_train, name='bn5'))
            # conv6 = self.new_conv_layer(bn5, [4,4,512,4000], stride=2, padding='VALID', name='conv6')
            # bn6 = self.leaky_relu(self.batchnorm(conv6, is_train, name='bn6'))

            # deconv4 = self.new_deconv_layer( bn6, [4,4,512,4000], conv5.get_shape().as_list(), padding='VALID', stride=2, name="deconv4")
            # debn4 = tf.nn.relu(self.batchnorm(deconv4, is_train, name='debn4'))
            # deconv3 = self.new_deconv_layer( debn4, [4,4,256,512], conv4.get_shape().as_list(), stride=2, name="deconv3")
            # debn3 = tf.nn.relu(self.batchnorm(deconv3, is_train, name='debn3'))
            # deconv2 = self.new_deconv_layer( debn3, [4,4,128,256], conv3.get_shape().as_list(), stride=2, name="deconv2")
            # debn2 = tf.nn.relu(self.batchnorm(deconv2, is_train, name='debn2'))
            # deconv1 = self.new_deconv_layer( debn2, [4,4,64,128], conv2.get_shape().as_list(), stride=2, name="deconv1")
            # debn1 = tf.nn.relu(self.batchnorm(deconv1, is_train, name='debn1'))
            # recon = self.new_deconv_layer( debn1, [4,4,3,64], [batch_size,64,64,3], stride=2, name="recon")

            # encoder

            previousFeatureMap = images
            previousDepth = 3
            depth = 64

            for layer in range(1, encoderLayerNum):
                print("build_reconstruction encoder layer=", layer)
                conv = self.new_conv_layer(previousFeatureMap, [4,4,previousDepth,depth], stride=2, name=("conv" + str(layer)))
                bn = self.leaky_relu(self.batchnorm(conv, is_train, name=("bn" + str(layer))))
                previousFeatureMap = bn
                previousDepth = depth
                depth = depth * 2

            # last layer
            conv = self.new_conv_layer(previousFeatureMap, [4,4,previousDepth,4000], stride=2, padding='VALID', name=('conv' + str(encoderLayerNum)))
            bn = self.leaky_relu(self.batchnorm(conv, is_train, name=("bn" + str(encoderLayerNum))))

            # decoder

            previousDepth = 4000
            depth = 64 * pow(2,decoderLayerNum-2)
            featureMapSize = 4

            deconv = self.new_deconv_layer( bn, [4,4,depth,previousDepth], [self.batch_size,featureMapSize,featureMapSize,depth], padding='VALID', stride=2, name=("deconv" + str(decoderLayerNum)))
            debn = tf.nn.relu(self.batchnorm(deconv, is_train, name=("debn" + str(decoderLayerNum))))

            previousFeatureMap = debn

            previousDepth = int(depth)
            depth = int(depth / 2)
            featureMapSize = int(featureMapSize *2)

            for layer in range(decoderLayerNum-1,1, -1):
                print("build_reconstruction decoder layer=", layer)
                deconv = self.new_deconv_layer( previousFeatureMap, [4,4,depth,previousDepth], [self.batch_size,featureMapSize,featureMapSize,depth], stride=2, name=("deconv" + str(layer)))
                debn = tf.nn.relu(self.batchnorm(deconv, is_train, name=('debn'+ str(layer))))
                previousFeatureMap = debn
                previousDepth = int(depth)
                depth = int(depth / 2)
                featureMapSize = int(featureMapSize *2)

            recon = self.new_deconv_layer( debn, [4,4,3,previousDepth], [self.batch_size,self.hiding_size,self.hiding_size,3], stride=2, name="recon")

        # return bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, recon, tf.nn.tanh(recon)
        return recon, tf.nn.tanh(recon)

    def build_adversarial(self, images, is_train, reuse=None):
        """
        Builds the dicriminator network
        """
        with tf.variable_scope('DIS', reuse=reuse):
            # conv1 = self.new_conv_layer(images, [4,4,3,64], stride=2, name="conv1" )
            # bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1'))
            # conv2 = self.new_conv_layer(bn1, [4,4,64,128], stride=2, name="conv2")
            # bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2'))
            # conv3 = self.new_conv_layer(bn2, [4,4,128,256], stride=2, name="conv3")
            # bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3'))
            # conv4 = self.new_conv_layer(bn3, [4,4,256,512], stride=2, name="conv4")
            # bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4'))

            # output = self.new_fc_layer( bn4, output_size=1, name='output')

            encoderLayerNum = self.encoderLayerNum

            previousFeatureMap = images
            previousDepth = 3
            depth = 64

            for layer in range(1, encoderLayerNum):
                print("build_adversarial encoder layer=", layer)
                conv = self.new_conv_layer(previousFeatureMap, [4,4,previousDepth,depth], stride=2, name=("conv" + str(layer)))
                bn = self.leaky_relu(self.batchnorm(conv, is_train, name=("bn" + str(layer))))
                previousFeatureMap = bn
                previousDepth = depth
                depth = depth * 2

            output = self.new_fc_layer( bn, output_size=1, name='output')

        return output[:,0]

