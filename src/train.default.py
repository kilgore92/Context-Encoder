#import ipdb
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model import *
from util import *
import pickle
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

n_epochs = 10000
learning_rate_val = 0.0003
weight_decay_rate =  0.00001
batch_size = 64 #500
lambda_recon = 0.9
lambda_adv = 0.1

image_size = 64
overlap_size = 7
hiding_size = 32

trainset_path = 'train'
testset_path  = 'test'
dataset_path = '/home/ibhat/image_completion/dcgan-completion.tensorflow/data/celebA'
model_path = '../models/celebA_ce_32'
result_path= '../results/celebA_ce_32'
pretrained_model_path = None

if os.path.exists(model_path):
    shutil.rmtree(model_path)

os.makedirs( model_path )

if not os.path.exists(result_path):
    os.makedirs( result_path )

if not os.path.exists( trainset_path ) or not os.path.exists( testset_path ):
    train_images = []
    for dir, _, _, in os.walk(dataset_path):
        train_images.extend( glob( os.path.join(dir, '*.jpg')))

    train_images = np.hstack(train_images)

    trainset = pd.DataFrame({'image_path':train_images[:int(len(train_images)*0.9)]})
    testset = pd.DataFrame({'image_path':train_images[int(len(train_images)*0.9):]})

testset.index = range(len(testset))
testset = testset.ix[np.random.permutation(len(testset))]
is_train = tf.placeholder( tf.bool )

learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, shape = [None, image_size, image_size, 3], name="images")

labels_D = tf.concat([tf.ones([batch_size]), tf.zeros([batch_size])],0)
labels_G = tf.ones([batch_size])
images_hiding = tf.placeholder( tf.float32, [None, hiding_size, hiding_size, 3], name='images_hiding') #Placeholder for patches

model = Model(image_size, hiding_size, batch_size)


writer = tf.summary.FileWriter("./../logs")

reconstruction_ori, reconstruction = model.build_reconstruction(images_tf, is_train)

adversarial_pos = model.build_adversarial(images_hiding, is_train)
adversarial_neg = model.build_adversarial(reconstruction, is_train, reuse=True) #D(F((1-M)*x))
adversarial_all = tf.concat([adversarial_pos, adversarial_neg],0)

# Applying bigger loss for overlapping region
mask_recon = tf.pad(tf.ones([hiding_size - 2*overlap_size, hiding_size - 2*overlap_size]), [[overlap_size,overlap_size], [overlap_size,overlap_size]])
mask_recon = tf.reshape(mask_recon, [hiding_size, hiding_size, 1])
mask_recon = tf.concat([mask_recon]*3,2)
mask_overlap = 1 - mask_recon

loss_recon_ori = tf.square( images_hiding - reconstruction )
loss_recon_center = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_recon, [1,2,3]))) / 10.  # Loss for non-overlapping region
loss_recon_overlap = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_overlap, [1,2,3]))) # Loss for overlapping region
loss_recon = loss_recon_center + loss_recon_overlap

loss_adv_D = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = adversarial_all, labels = labels_D))
loss_adv_G = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = adversarial_neg, labels = labels_G))


# G : Encoder-decoder structure
# The weights of this structure contribute to both the adv and L2 loss,
# Thus, loss_G is a weighted sum of these
loss_G = loss_adv_G * lambda_adv + loss_recon * lambda_recon
# The discriminator is trained on the standard GAN loss
loss_D = loss_adv_D * lambda_adv

var_G = list(filter( lambda x: x.name.startswith('GEN'), tf.trainable_variables()))
var_D = list(filter( lambda x: x.name.startswith('DIS'), tf.trainable_variables()))

W_G = list(filter(lambda x: x.name.endswith('W:0'), var_G))
W_D = list(filter(lambda x: x.name.endswith('W:0'), var_D))

# Regularization
loss_G += weight_decay_rate * tf.reduce_mean(tf.stack( list(map(lambda x: tf.nn.l2_loss(x), W_G))))
loss_D += weight_decay_rate * tf.reduce_mean(tf.unstack( list(map(lambda x: tf.nn.l2_loss(x), W_D))))

loss_adv_G_sum = tf.summary.scalar("loss_adv_G", loss_adv_G)
loss_recon_sum = tf.summary.scalar("loss_recon", loss_G)
loss_adv_D_sum = tf.summary.scalar("loss_adv_D", loss_G)
g_loss_sum = tf.summary.scalar("loss_G", loss_G)
d_loss_sum = tf.summary.scalar("loss_D", loss_D)
g_img_sum = tf.summary.image("G", reconstruction)

g_summary = tf.summary.merge([loss_adv_G_sum, loss_recon_sum, g_loss_sum, g_img_sum])
d_summary = tf.summary.merge([loss_adv_D_sum, d_loss_sum])

sess = tf.InteractiveSession()

optimizer_G = tf.train.AdamOptimizer( learning_rate=learning_rate )
grads_vars_G = optimizer_G.compute_gradients( loss_G, var_list=var_G )
grads_vars_G = list(map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_G))
train_op_G = optimizer_G.apply_gradients( grads_vars_G )

optimizer_D = tf.train.AdamOptimizer( learning_rate=learning_rate )
grads_vars_D = optimizer_D.compute_gradients( loss_D, var_list=var_D )
grads_vars_D = list(map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_D))
train_op_D = optimizer_D.apply_gradients( grads_vars_D )

saver = tf.train.Saver(max_to_keep=1)

init = tf.global_variables_initializer()
sess.run(init)

if pretrained_model_path is not None and os.path.exists( pretrained_model_path ):
    saver.restore( sess, pretrained_model_path )

iters = 0

loss_D_val = 0.
loss_G_val = 0.

crop_pos = (image_size - hiding_size)/2

for epoch in range(n_epochs):
    trainset.index = range(len(trainset))
    trainset = trainset.ix[np.random.permutation(len(trainset))]

    for start,end in zip(
            range(0, len(trainset), batch_size),
            range(batch_size, len(trainset), batch_size)):


        image_paths = trainset[start:end]['image_path'].values # Selects a batch randomly
        images_ori = list(map(lambda x: load_image( x ,pre_width=(image_size+18), pre_height=(image_size+18),width=image_size,height=image_size), image_paths))
        is_none = np.sum(list(map(lambda x: x is None, images_ori)))
        if (is_none > 0): continue

        # images_crops = map(lambda x: crop_random(x,width=hiding_size,height=hiding_size, overlap=overlap_size), images_ori)
        images_crops = list(map(lambda x: crop_random(x,width=hiding_size,height=hiding_size,x=crop_pos, y=crop_pos, overlap=overlap_size, isShake=True), images_ori))
        images, crops,_,_ = zip(*images_crops) #Unpack returned values

        # Printing activations every 100 iterations
        if iters % 100 == 0:
            test_image_paths = testset[:batch_size]['image_path'].values
            test_images_ori = list(map(lambda x: load_image(x,pre_width=(image_size+18), pre_height=(image_size+18),width=image_size,height=image_size), test_image_paths))

            test_images_crop = list(map(lambda x: crop_random(x, width=hiding_size,height=hiding_size, x=crop_pos, y=crop_pos, overlap=overlap_size), test_images_ori))
            test_images, test_crops, xs,ys = zip(*test_images_crop)

            # reconstruction_vals, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val, loss_G_val, loss_D_val = sess.run(
            #         [reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1, loss_G, loss_D],
            #         feed_dict={
            #             images_tf: test_images,
            #             images_hiding: test_crops,
            #             is_train: False
            #             })

            reconstruction_vals, recon_ori_vals, loss_G_val, loss_D_val = sess.run(
                    [reconstruction, reconstruction_ori, loss_G, loss_D],
                    feed_dict={
                        images_tf: test_images,
                        images_hiding: test_crops,
                        is_train: False
                        })

            # Generate result every 500 iterations
            if iters % 500 == 0:
                ii = 0
                for rec_val, img,x,y in zip(reconstruction_vals, test_images, xs, ys):
                    rec_hid = (255. * (rec_val+1)/2.).astype(int)
                    rec_con = (255. * (img+1)/2.).astype(int)

                    rec_con[int(y):int(y+hiding_size), int(x):int(x+hiding_size)] = rec_hid
                    cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.'+str(int(iters/100))+'.jpg'), rec_con)
                    ii += 1
                    if ii > 50: break

                if iters == 0:
                    ii = 0
                    for test_image in test_images_ori:
                        test_image = (255. * (test_image+1)/2.).astype(int)
                        test_image[int(crop_pos):int(crop_pos+hiding_size),int(crop_pos):int(crop_pos+hiding_size)] = 0
                        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.ori.jpg'), test_image)
                        ii += 1
                        if ii > 50: break

            print ("========================================================================")
            print (recon_ori_vals.max(), recon_ori_vals.min())
            print (reconstruction_vals.max(), reconstruction_vals.min())
            print (loss_G_val, loss_D_val)
            print ("=========================================================================")

            if np.isnan(reconstruction_vals.min() ) or np.isnan(reconstruction_vals.max()):
                print ("NaN detected!!")
                #ipdb.set_trace()

        # Generative Part (Enc-Dec) + Discriminator is updated every iteration (Alternating SGD)

        _, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_G_val, reconstruction_vals, recon_ori_vals, g_summary_str = sess.run(
                [train_op_G, loss_G, adversarial_pos, adversarial_neg, loss_recon, loss_adv_G, reconstruction, reconstruction_ori,g_summary],
                feed_dict={
                    images_tf: images, # image with missing patch
                    images_hiding: crops, #patches
                    learning_rate: learning_rate_val,
                    is_train: True
                    })

        writer.add_summary(g_summary_str, iters)

        _, loss_D_val, adv_pos_val, adv_neg_val, d_summary_str = sess.run(
                [train_op_D, loss_D, adversarial_pos, adversarial_neg,d_summary],
                feed_dict={
                    images_tf: images,
                    images_hiding: crops,
                    learning_rate: learning_rate_val/10.,
                    is_train: True
                        })

        writer.add_summary(d_summary_str, iters)

        # _, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_G_val, reconstruction_vals, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val = sess.run(
        #         [train_op_G, loss_G, adversarial_pos, adversarial_neg, loss_recon, loss_adv_G, reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1],
        #         feed_dict={
        #             images_tf: images,
        #             images_hiding: crops,
        #             learning_rate: learning_rate_val,
        #             is_train: True
        #             })


        # _, loss_D_val, adv_pos_val, adv_neg_val = sess.run(
        #         [train_op_D, loss_D, adversarial_pos, adversarial_neg],
        #         feed_dict={
        #             images_tf: images,
        #             images_hiding: crops,
        #             learning_rate: learning_rate_val/10.,
        #             is_train: True
        #                 })

        print ("Iter:: {} , Gen Loss: {}, Recon Loss: {}, Gen ADV Loss: {}  Dis Loss: {}".format(iters,loss_G_val,loss_recon_val,loss_adv_G_val,loss_D_val))

        iters += 1


    saver.save(sess,os.path.join(model_path,'model.ckpt'), global_step=epoch)
    learning_rate_val *= 0.99



