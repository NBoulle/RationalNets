"""

@author: Nicolas Boulle
Code modified from https://keras.io/examples/mnist_acgan/

GAN experiment on MNIST of the paper
Run the following command to use a rational network:
python3 mnist_gan.py --rational=True

or a ReLU network:
python3 mnist_gan.py --rational=False

"""

from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization, ReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
import argparse

# Load the Rational Layer
import os, sys
sys.path.insert(0, '..')
from RationalLayer import RationalLayer

np.random.seed(1337)
num_classes = 10

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    # Coefficients of the rational approximating ReLU
    alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218]
    beta_initializer = [2.383, 0.0, 1.0]
    if UseRational:
        cnn = Sequential()
        cnn.add(Dense(3 * 3 * 384, input_dim=latent_size))
        cnn.add(RationalLayer(alpha_initializer, beta_initializer, shared_axes=[1]))
        cnn.add(Reshape((3, 3, 384)))
        # upsample to (7, 7, ...)
        cnn.add(Conv2DTranspose(192, 5, strides=1, padding='valid',
                                kernel_initializer='glorot_normal'))
        cnn.add(RationalLayer(alpha_initializer, beta_initializer, shared_axes=[1,2,3]))
        cnn.add(BatchNormalization())
        # upsample to (14, 14, ...)
        cnn.add(Conv2DTranspose(96, 5, strides=2, padding='same',
                                kernel_initializer='glorot_normal'))
        cnn.add(RationalLayer(alpha_initializer, beta_initializer, shared_axes=[1,2,3]))
        cnn.add(BatchNormalization())
        # upsample to (28, 28, ...)
        cnn.add(Conv2DTranspose(1, 5, strides=2, padding='same',
                                activation='tanh',
                                kernel_initializer='glorot_normal'))
    else:
        cnn = Sequential()
        cnn.add(Dense(3 * 3 * 384, input_dim=latent_size))
        cnn.add(ReLU())
        cnn.add(Reshape((3, 3, 384)))
        # upsample to (7, 7, ...)
        cnn.add(Conv2DTranspose(192, 5, strides=1, padding='valid',
                                kernel_initializer='glorot_normal'))
        cnn.add(ReLU())
        cnn.add(BatchNormalization())
        # upsample to (14, 14, ...)
        cnn.add(Conv2DTranspose(96, 5, strides=2, padding='same',
                                kernel_initializer='glorot_normal'))
        cnn.add(ReLU())
        cnn.add(BatchNormalization())
        # upsample to (28, 28, ...)
        cnn.add(Conv2DTranspose(1, 5, strides=2, padding='same',
                                activation='tanh',
                                kernel_initializer='glorot_normal'))

    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    cls = Embedding(num_classes, latent_size,
                    embeddings_initializer='glorot_normal')(image_class)

    # hadamard product between z-space and a class conditional embedding
    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

    return Model([latent, image_class], fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    # Coefficients of the rational approximating ReLU
    alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218]
    beta_initializer = [2.383, 0.0, 1.0]
    if UseRational:
        cnn = Sequential()
        cnn.add(Conv2D(32, 3, padding='same', strides=2,
                       input_shape=(28, 28, 1)))
        cnn.add(RationalLayer(alpha_initializer, beta_initializer, shared_axes=[1,2,3]))
        cnn.add(Dropout(0.3))
        cnn.add(Conv2D(64, 3, padding='same', strides=1))
        cnn.add(RationalLayer(alpha_initializer, beta_initializer, shared_axes=[1,2,3]))
        cnn.add(Dropout(0.3))
        cnn.add(Conv2D(128, 3, padding='same', strides=2))
        cnn.add(RationalLayer(alpha_initializer, beta_initializer, shared_axes=[1,2,3]))
        cnn.add(Dropout(0.3))
        cnn.add(Conv2D(256, 3, padding='same', strides=1))
        cnn.add(RationalLayer(alpha_initializer, beta_initializer, shared_axes=[1,2,3]))
        cnn.add(Dropout(0.3))
        cnn.add(Flatten())
    else:
        cnn = Sequential()
        cnn.add(Conv2D(32, 3, padding='same', strides=2,
                       input_shape=(28, 28, 1)))
        cnn.add(LeakyReLU(0.2))
        cnn.add(Dropout(0.3))
        cnn.add(Conv2D(64, 3, padding='same', strides=1))
        cnn.add(LeakyReLU(0.2))
        cnn.add(Dropout(0.3))
        cnn.add(Conv2D(128, 3, padding='same', strides=2))
        cnn.add(LeakyReLU(0.2))
        cnn.add(Dropout(0.3))
        cnn.add(Conv2D(256, 3, padding='same', strides=1))
        cnn.add(LeakyReLU(0.2))
        cnn.add(Dropout(0.3))
        cnn.add(Flatten())

    image = Input(shape=(28, 28, 1))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)

    return Model(image, [fake, aux])


if __name__ == '__main__':
    
    # Use rational or ReLU network
    parser = argparse.ArgumentParser()
    parser.add_argument("--rational", type=bool, default = True)
    args, _ = parser.parse_known_args()
    UseRational = args.rational
    
    # batch and latent size taken from the paper
    epochs = 100
    batch_size = 100
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # Create path to save the results
    if UseRational:
        path = 'Results_rational'
    else:
        path = 'Results_relu'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + '/'
    
    # build the discriminator
    print('Discriminator model:')
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    discriminator.summary()

    # build the generator
    generator = build_generator(latent_size)

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model([latent, image_class], [fake, aux])

    print('Combined model:')
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    combined.summary()

    # get our mnist data, and force it to be of shape (..., 28, 28, 1) with
    # range [-1, 1]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=-1)

    num_train, num_test = x_train.shape[0], x_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    
    # Save examples of MNIST data
    # prepare real images sorted by class label
    num_rows = 40
    
    # Select 40 random indices per class label
    Index = np.array([])
    for i in range(num_classes):
        indice_i = np.argwhere(y_train==i)
        np.random.shuffle(indice_i)
        L = indice_i[:num_rows,0]
        Index = np.concatenate((Index, L))
    indices = [int(i) for i in Index]
    real_images = x_train[indices]
    img = real_images
    img = (np.concatenate([r.reshape(-1, 28)
                           for r in np.split(img, num_classes)
                           ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
    Image.fromarray(img).save('mnist_images.png')

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))

        num_batches = int(np.ceil(x_train.shape[0] / float(batch_size)))
        progress_bar = Progbar(target=num_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            # get a batch of real images
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (len(image_batch), latent_size))

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, num_classes, len(image_batch))

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (len(image_batch), 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            x = np.concatenate((image_batch, generated_images))

            # use one-sided soft real/fake labels
            # Salimans et al., 2016
            # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
            soft_zero, soft_one = 0, 0.95
            y = np.array(
                [soft_one] * len(image_batch) + [soft_zero] * len(image_batch))
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # we don't want the discriminator to also maximize the classification
            # accuracy of the auxiliary classifier on generated images, so we
            # don't train discriminator to produce class labels for generated
            # images (see https://openreview.net/forum?id=rJXTf9Bxg).
            # To preserve sum of sample weights for the auxiliary classifier,
            # we assign sample weight of 2 to the real images.
            disc_sample_weight = [np.ones(2 * len(image_batch)),
                                  np.concatenate((np.ones(len(image_batch)) * 2,
                                                  np.zeros(len(image_batch))))]

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(
                x, [y, aux_y], sample_weight=disc_sample_weight))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * len(image_batch), latent_size))
            sampled_labels = np.random.randint(0, num_classes, 2 * len(image_batch))

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * len(image_batch)) * soft_one

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels]))

            progress_bar.update(index + 1)

        print('Testing for epoch {}:'.format(epoch))

        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (num_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, num_classes, num_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        x = np.concatenate((x_test, generated_images))
        y = np.array([1] * num_test + [0] * num_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            x, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, 2 * num_test)

        trick = np.ones(2 * num_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f} | {3:<5.4f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(path+'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(path+'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        num_rows = 40
        noise = np.tile(np.random.uniform(-1, 1, (num_rows, latent_size)),
                        (num_classes, 1))

        sampled_labels = np.array([
            [i] * num_rows for i in range(num_classes)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)

        # prepare real images sorted by class label
        real_labels = y_train[(epoch - 1) * num_rows * num_classes:
                              epoch * num_rows * num_classes]
        indices = np.argsort(real_labels, axis=0)
        real_images = x_train[(epoch - 1) * num_rows * num_classes:
                              epoch * num_rows * num_classes][indices]
        
        img = generated_images
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(img, num_classes)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(path+'plot_epoch_{0:03d}_generated.png'.format(epoch))

    with open(path+'acgan-history.pkl', 'wb') as f:
        pickle.dump({'train': train_history, 'test': test_history}, f)