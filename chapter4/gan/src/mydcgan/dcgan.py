import os

from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, ReLU
from keras.layers import Reshape
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
import numpy as np
import cv2
import matplotlib.pyplot as plt


class MyDCGAN():
    def __init__(self):

        self.train_size = 100000
        self.shape = (64, 64, 3)  # 画像サイズ
        self.z_dim = 100            # 潜在変数の次元
        self.unrolling_steps = 5

        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.combined = self.combined_model(
            self.generator,
            self.discriminator
        )
        # d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        d_optim = Adam(lr=0.0002, beta_1=0.5)
        g_optim = Adam(lr=0.0002, beta_1=0.5)

        self.generator.compile(
            loss="binary_crossentropy",
            optimizer="Adam"
        )
        self.combined.compile(
            loss="binary_crossentropy",
            optimizer=g_optim
        )
        self.discriminator.trainable = True
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=d_optim,
            metrics=['accuracy']
        )

    def generator_model(self):
        noise_shape = (self.z_dim,)

        model = Sequential()

        model.add(Dense(128 * 8 * 8, input_shape=noise_shape))
        model.add(Activation("relu"))
        model.add(Reshape((8, 8, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(3, (1, 1), padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        return model

    def discriminator_model(self):
        img_shape = self.shape

        model = Sequential()

        model.add(Conv2D(32, (3, 3),
                         strides=2,
                         padding="same",
                         input_shape=img_shape)
                  )
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(64, (3, 3), strides=2, padding="same"))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, (3, 3), strides=2, padding="same"))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), strides=1, padding="same"))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        return model

    def cache_discriminator_weights(self):
        self.discriminator.save_weights(
            os.path.dirname(
                os.path.abspath(__file__)
            ) + "\\model\\discriminator",
                True
        )

    def restore_discriminator_weights(self):
        self.discriminator.load_weights(
            os.path.dirname(
                os.path.abspath(__file__)
            ) + "\\model\\discriminator"
        )

    def combined_model(self, generator, discriminator):
        model = Sequential()

        model.add(generator)
        discriminator.trainable = False
        model.add(discriminator)

        return model

    def load_imgs(self, idx):
        img_paths = []
        for i in idx:
            img_paths.append(
                os.path.dirname(
                    os.path.abspath(__file__)
                ) + "\\dataset64\\img_" + str(i) + ".png"
            )

        images = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        images = np.array(images)

        return np.array(images)

    def save_imgs(self, epoch, index, check_noise, r, c):
        noise = check_noise
        gen_imgs = self.generator.predict(noise)

        # 0-1 rescale
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        path = os.path.dirname(
            os.path.abspath(__file__)
        ) + "\\images\\" + str(epoch) + "_" + str(index) + ".png"
        fig.savefig(path)

        plt.close()

    def train(self, batch_size=32, n_epoch=100, check_noise=None, r=5, c=5):

        half_batch = int(batch_size / 2)

        noise = np.zeros((half_batch, self.z_dim))

        for epoch in range(n_epoch):
            shuffle_idx = np.random.permutation(list(range(self.train_size)))

            print("Epoch is", epoch)
            print(
                "Number of batches", int(
                    self.train_size / half_batch - self.unrolling_steps))
            for index in range(
                    int(self.train_size / half_batch - self.unrolling_steps)):
                # -------- 識別モデルの学習 -------------------------------------------
                # 数回先に学習する
                for unroll_index in range(index, index + self.unrolling_steps):
                    unroll_train = self.load_imgs(
                        shuffle_idx[unroll_index * half_batch:(unroll_index + 1) * half_batch]
                    )

                    unroll_train = (
                        unroll_train.astype(np.float32) - 127.5) / 127.5

                    for i in range(half_batch):
                        noise[i, :] = np.random.uniform(-1, 1, self.z_dim)

                    image_batch = unroll_train

                    generated_images = self.generator.predict(noise, verbose=0)
                    if index % 50 == 0:
                        self.save_imgs(epoch, index, check_noise, r, c)

                    # X = np.concatenate((image_batch, generated_images), axis=0)
                    # y = [1] * half_batch + [0] * half_batch
                    # d_loss = self.discriminator.train_on_batch(X, y)

                    d_loss_real = self.discriminator.train_on_batch(
                        image_batch,
                        [1] * half_batch
                    )
                    d_loss_fake = self.discriminator.train_on_batch(
                        generated_images,
                        [0] * half_batch
                    )

                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    if unroll_index == index:
                        self.cache_discriminator_weights()
                        print("batch %d d_loss : %f" % (index, d_loss[0]))

                # -------- 生成モデルの学習-------------------------------------------
                for i in range(half_batch):
                    noise[i, :] = np.random.uniform(-1, 1, self.z_dim)

                self.discriminator.trainable = False
                g_loss = self.combined.train_on_batch(
                    noise,
                    [1] * half_batch
                )
                self.discriminator.trainable = True
                print("batch %d g_loss : %f" % (index, g_loss))
                self.restore_discriminator_weights()

        self.discriminator.save_weights(
            os.path.dirname(
                os.path.abspath(__file__)
            ) + "\\model\\discriminator",
                True
        )
        self.generator.save_weights(
            os.path.dirname(
                os.path.abspath(__file__)
            ) + "\\model\\generator",
                True
        )


if __name__ == "__main__":
    dcgan = MyDCGAN()

    r, c = 4, 4
    check_noise = np.random.uniform(-1, 1, (r * c, 100))
    dcgan.train(
        batch_size=512,
        n_epoch=50,
        check_noise=check_noise,
        r=r,
        c=c
    )
