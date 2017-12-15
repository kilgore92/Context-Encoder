import model
import tensorflow as tf
import os
import random
import cv2
from util import *
import shutil

def create_file_list(image_dir,num_samples=50,sample=True):

    """
    Creates list of file names of all images in the dataset

    """
    data_files = (os.listdir(image_dir))
    files = [os.path.join(image_dir,f) for f in data_files]
    if sample:
        sample_files = random.sample(files,num_samples)
        return sample_files
    else:
        return files

def load(sess,saver,checkpoint_dir):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    else:
        return False

dataset_path = '/home/ibhat/image_completion/dcgan-completion.tensorflow/data/celebA'
test_image_files = create_file_list(image_dir = dataset_path,num_samples=5)
checkpoint_path = '/home/ibhat/context_enc/Context-Encoder/models/celebA_vanilla'
dump_dir = 'test_dump'

if os.path.exists(dump_dir):
    shutil.rmtree(dump_dir)

os.makedirs(dump_dir)

#Instantiate model
image_size = 64
hiding_size = 16
batch_size = 1
overlap_size = 7

model = model.Model(image_size = image_size , hiding_size = hiding_size , batch_size = batch_size )
images_tf = tf.placeholder( tf.float32, [None, image_size, image_size, 3], name="images")
images_hiding = tf.placeholder( tf.float32, [None, hiding_size, hiding_size, 3], name='images_hiding') #Placeholder for patches
is_train = tf.placeholder( tf.bool )

crop_pos = (image_size - hiding_size)/2

# Define the comp nodes of the TF graph that we need
reconstruction_ori, reconstruction = model.build_reconstruction(images_tf, is_train)

#Start the session
sess = tf.InteractiveSession()

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

isLoaded = load(sess = sess ,saver = saver,checkpoint_dir=checkpoint_path)
print(isLoaded)

for test_image_path,idx in zip(test_image_files,range(len(test_image_files))):
    test_image = load_image(test_image_path,pre_width=(image_size+18), pre_height=(image_size+18),width=image_size,height=image_size)
    cropped_image,crop,xs,ys = crop_random(test_image, width=hiding_size,height=hiding_size, x=crop_pos, y=crop_pos, overlap=overlap_size)

    reconstruction_vals,reconstruction_ori_vals  = sess.run(
            [reconstruction,reconstruction_ori],
            feed_dict={
                images_tf: cropped_image.reshape(1,image_size,image_size,3),
                images_hiding: crop.reshape(1,hiding_size,hiding_size,3),
                is_train: False
                })

    # Missing patch
    recon_missing = (255.*(reconstruction_ori_vals+1)/2.).astype(int)
    true_image = (255. * (cropped_image + 1)/2.).astype(int)
    patch = (255. * (crop + 1)/2.).astype(int)
    true_image_filled = true_image.copy()
    print('{} {}'.format(xs,ys))
    true_image_filled[int(crop_pos):int(crop_pos+hiding_size), int(crop_pos):int(crop_pos+hiding_size)] = recon_missing
    cv2.imwrite(os.path.join(dump_dir,'ori_{}.jpg'.format(idx)),true_image)
    cv2.imwrite(os.path.join(dump_dir,'recon_{}.jpg'.format(idx)),true_image_filled)
    cv2.imwrite(os.path.join(dump_dir,'patch_{}.jpg'.format(idx)),patch)
    print('Saved image {}'.format(idx))



