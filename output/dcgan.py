# sean sungil kim

from __future__ import print_function

import argparse
import os
import numpy as np
from PIL import Image
import math
import glob
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Reshape, LeakyReLU, Dropout, Conv2DTranspose
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from cv2 import resize
from sklearn.preprocessing import MinMaxScaler
import scipy.misc


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, default = 'train')
    parser.add_argument('--image_size', type = int, default = 96)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_epoch', type = int, default = 10000)
    parser.add_argument('--pretty', dest = 'pretty', action = 'store_true')
    parser.add_argument('--image_dir', type = str, default = "D:/SIK/Project Bolt/GAN_images")
    parser.add_argument('--txt_dir', type = str, default = "D:/SIK/Project Bolt/preprocessed_spec_graphs_1_wT/*.txt")
    parser.set_defaults(pretty = False)
    args = parser.parse_args()
    
    return args


def generator_model():
    
    model = Sequential()
    model.add(Dense(12*9*256, input_shape = (96, )))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((12,9,256), input_shape = (12*9*256, )))

    model.add(Conv2DTranspose(128, (5, 5), strides = (2, 2),
                                           padding = 'same',
                                           use_bias = False,
                                           data_format = "channels_last",))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides = (2, 2),
                                          padding = 'same',
                                          use_bias = False,
                                          data_format = "channels_last"))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides = (2, 2),
                                         padding = 'same',
                                         use_bias = False,
                                         data_format = "channels_last", 
                                         activation = 'tanh'))
    
    return model


def discriminator_model(image_size):
    
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides = (2, 2),
                                 padding = 'same',
                                 data_format = "channels_last",
                                 input_shape = (96, 72, 1)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides = (2, 2),
                                  padding = 'same',
                                  data_format = "channels_last"))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(32))
    model.add(Flatten())
    model.add(Dense(1))
    
    return model


def generator_containing_discriminator(generator, discriminator):
    
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    
    return model


def data_load(image_size, txt_dir):

    txt_lst = glob.glob(txt_dir)
    
    cnt = 0
    for txt_file in txt_lst:
        if cnt == 0:
            data = np.expand_dims(np.expand_dims(resize(np.loadtxt(txt_file), (72, 96)), axis = 0), axis = 3)
            cnt += 1
        else:
            data = np.concatenate([data, np.expand_dims(np.expand_dims(resize(np.loadtxt(txt_file), (72, 96)), axis = 0), axis = 3)], axis = 0)

    data_scaled = MinMaxScaler(feature_range = (0, 1)).fit_transform(data.reshape(-1, 1)).reshape(data.shape)
    
    return data_scaled


def train(image_size, batch_size, num_epoch, txt_dir, image_dir):

    X_train = data_load(image_size, txt_dir)
    discriminator = discriminator_model(image_size)
    generator = generator_model()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    d_optim = Adam(0.0002, 0.5)
    g_optim = Adam(0.0002, 0.5)
    generator.compile(loss = 'binary_crossentropy', optimizer = "SGD")
    discriminator_on_generator.compile(loss = 'binary_crossentropy', optimizer = g_optim)
    discriminator.trainable = True
    discriminator.compile(loss = 'binary_crossentropy', optimizer = d_optim)
    noise = np.zeros((batch_size, 96))
    
    for epoch in range(num_epoch):
        print("Epoch " + str(epoch + 1) + "/" + str(num_epoch) + ", " + str(int(X_train.shape[0] / batch_size)) + " batches:")
        
        for index in range(int(X_train.shape[0] / batch_size)):
            for i in range(batch_size):
                noise[i, :] = np.random.normal(-1, 1, 96)
                
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            generated_images = generator.predict(noise, verbose = 0)
            
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X, y)
            print("Batch %d d_loss : %f" % (index + 1, d_loss))
            
            for i in range(batch_size):
                noise[i, :] = np.random.normal(-1, 1, 96)
                
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * batch_size)
            discriminator.trainable = True
            print("Batch %d g_loss : %f" % (index + 1, g_loss))
            
        generator.save_weights(image_dir + '/' + 'generator_weights_' + str(epoch), True)
        discriminator.save_weights(image_dir + '/' + 'discriminator_weights_' + str(epoch), True)
        
        if epoch % 3 == 0:
            for i in range(generated_images.shape[0]):
                gen_img = generated_images[i].reshape(generated_images[i].shape[0], generated_images[i].shape[1])

                if len(str(epoch)) != 4:
                    scipy.misc.imsave(image_dir + '/' + ('0'*(4-len(str(epoch + 1)))) + str(epoch + 1) + '_' + str(i) + '.png', gen_img)
                else:
                    scipy.misc.imsave(image_dir + '/' + str(epoch + 1) + '_' + str(i) + '.png', gen_img)
                    
                    
'''
def generate(image_size, batch_size, pretty, image_dir):
    
    generator = generator_model()
    generator.compile(loss = 'binary_crossentropy', optimizer = "SGD")
    generator.load_weights('generator_weights')
    
    if pretty:
        discriminator = discriminator_model(image_size)
        discriminator.compile(loss = 'binary_crossentropy', optimizer = "SGD")
        discriminator.load_weights('discriminator_weights')
        noise = np.zeros((batch_size * 20, 96))
        
        for i in range(batch_size * 20):
            noise[i, :] = np.random.normal(-1, 1, 96)
            
        generated_images = generator.predict(noise, verbose = 1)
        d_pret = discriminator.predict(generated_images, verbose = 1)
        index = np.arange(0, batch_size * 20)
        index.resize((batch_size * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis = 1))
        pre_with_index.sort(key = lambda x: x[0], reverse = True)
        pretty_images = np.zeros((batch_size, 1) + (generated_images.shape[2:]), dtype = np.float32)
        
        for i in range(int(batch_size)):
            idx = int(pre_with_index[i][1])
            pretty_images[i, 0, :, :] = generated_images[idx, 0, :, :]
            
        image = combine_images(pretty_images)
    else:
        noise = np.zeros((batch_size, 96))
        
        for i in range(batch_size):
            noise[i, :] = np.random.normal(-1, 1, 96)
            
        generated_images = generator.predict(noise, verbose = 1)
        image = combine_images(generated_images)
        
    Image.fromarray(image.astype(np.uint8)).save(image_dir + "/generated_image.png")
'''


def main():
    
    args = get_args()
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    if args.mode == 'train':
        train(image_size = args.image_size, 
              batch_size = args.batch_size, 
              num_epoch = args.num_epoch, 
              txt_dir = args.txt_dir,
              image_dir = args.image_dir)
    elif args.mode == 'generate':
        generate(image_size = args.image_size, 
                 batch_size = args.batch_size, 
                 pretty = args.pretty, 
                 image_dir = args.image_dir)
    
    print("\n================ Finished ================\n")

if __name__ == "__main__":
    main()
