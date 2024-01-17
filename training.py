import cv2 as cv2
import os
import tensorflow as tf
import keras
import matplotlib
import numpy as np
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

dataset_path = r'D:/IMAGE_FORGERY_DATASET/Dataset'
forged_path = r'D:/IMAGE_FORGERY_DATASET/Dataset/Forged'
real_path = r'D:/IMAGE_FORGERY_DATASET/Dataset/Original'
img_height, img_width = 128,128
batch_size = 32

def load_images_from_directory_real(directory, target_size=(img_height, img_width)):
    images = []
    labels = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        label = 1 
        img = cv2.imread(path)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)
real_images,real_labels = load_images_from_directory_real(real_path, target_size=(img_height, img_width))
print(real_images,real_labels)

def load_images_from_directory_forged(directory, target_size=(img_height, img_width)):
    images = []
    labels = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        label = 0 
        img = cv2.imread(path)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)
forged_images,forged_labels = load_images_from_directory_forged(forged_path, target_size=(img_height, img_width))
print(forged_images,forged_labels)

print(real_images.shape)
print(forged_images.shape)

import numpy as np

images = []
labels = []
images.append(real_images)
labels.append(real_labels)
images.append(forged_images)
labels.append(forged_labels)
labels = np.concatenate(labels, axis=0)
images = np.concatenate(images, axis=0)
print(labels)
print(images.shape)

from keras import layers, models


def build_generator(latent_dim=100, img_height=128, img_width=128):
    model = models.Sequential()
    model.add(layers.Dense(128 * 16 * 16, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((16, 16, 128)))
    target_shape = (img_height // 4, img_width // 4)  # Target size after upsampling
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid'))
    # Adjust channels to 3 for RGB
    return model

def build_discriminator(img_height=128, img_width=128):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_gan_model(generator, discriminator, latent_dim=100):
    discriminator.trainable = False  # Freeze the discriminator during combined model training
    gan_input = layers.Input(shape=(latent_dim,))
    generated_image = generator(gan_input)
    print(generated_image.shape)
    gan_output = discriminator(generated_image)
    gan_model = models.Model(gan_input, gan_output)
    gan_model.compile(loss='binary_crossentropy', optimizer='adam')
    return gan_model
# Example usage

generator = build_generator()
discriminator = build_discriminator()
gan_model = build_gan_model(generator, discriminator)

epochs = 50

discriminator = build_discriminator()
generator = build_generator()
steps_per_epoch = min(len(real_images), len(forged_images)) // batch_size

gan_model = build_gan_model(generator, discriminator)

discriminator.fit(images,labels,epochs = 25,verbose = 1)





