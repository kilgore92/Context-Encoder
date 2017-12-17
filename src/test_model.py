import tensorflow as tf
from model import *
import os
import random
import cv2
from util import *
import shutil
import pickle

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
        print(ckpt.model_checkpoint_path)
        return True
    else:
        return False

dataset_path = '/home/ibhat/image_completion/dcgan-completion.tensorflow/data/celebA'
checkpoint_path = '/home/ibhat/context_enc/Context-Encoder/models/celebA_ce'
dump_dir = 'test_dump'

if os.path.exists(dump_dir):
    shutil.rmtree(dump_dir)

os.makedirs(dump_dir)

#Instantiate model
image_size = 64
hiding_size = 16
batch_size = 64
overlap_size = 7

num_batches = 8
test_image_files = create_file_list(image_dir = dataset_path,num_samples=num_batches*batch_size)
with open('test_images.pkl','wb') as f:
    pickle.dump(test_image_files,f)

model = Model(image_size = image_size , hiding_size = hiding_size , batch_size = batch_size )
images_tf = tf.placeholder( tf.float32, [None, image_size, image_size, 3], name="images")
images_hiding = tf.placeholder( tf.float32, [None, hiding_size, hiding_size, 3], name='images_hiding') #Placeholder for patches
is_train = tf.placeholder( tf.bool )

crop_pos = 24

# Define the comp nodes of the TF graph that we need
reconstruction_ori, reconstruction = model.build_reconstruction(images_tf, is_train)

#Start the session
sess = tf.InteractiveSession()

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

isLoaded = load(sess = sess ,saver = saver,checkpoint_dir=checkpoint_path)
print(isLoaded)

idx = 0 # Used to number the folders created

for batch in range(num_batches):
    test_images_batch = test_image_files[batch*batch_size:(batch*batch_size+batch_size)]
    test_images_ori = list(map(lambda x: load_image(x,pre_width=(image_size+18), pre_height=(image_size+18),width=image_size,height=image_size), test_images_batch))
    test_images_crop = list(map(lambda x: crop_random(x, width=hiding_size,height=hiding_size, x=crop_pos, y=crop_pos, overlap=overlap_size), test_images_ori))
    test_images, test_crops, xs,ys = zip(*test_images_crop)

    reconstruction_vals,reconstruction_ori_vals  = sess.run(
            [reconstruction,reconstruction_ori],
            feed_dict={
                images_tf: test_images,
                images_hiding: test_crops,
                is_train: False
                })

    for recon_patch,image in zip(reconstruction_vals,test_images):
        save_dir = os.path.join(dump_dir,'{}'.format(idx))
        os.makedirs(save_dir)
        recon_missing = (255. *(recon_patch+1)/2.).astype(int)
        true_image = (255. * (image + 1)/2.).astype(int)
        true_image_filled = true_image.copy()
        true_image_missing = true_image.copy()
        true_image_missing[int(crop_pos):int(crop_pos+hiding_size), int(crop_pos):int(crop_pos+hiding_size)] = 0
        true_image_filled[int(crop_pos):int(crop_pos+hiding_size), int(crop_pos):int(crop_pos+hiding_size)] = recon_missing
        cv2.imwrite(os.path.join(save_dir,'ori.jpg'),true_image)
        cv2.imwrite(os.path.join(save_dir,'recon.jpg'),true_image_filled)
        cv2.imwrite(os.path.join(save_dir,'masked.jpg'),true_image_missing)
        idx += 1


