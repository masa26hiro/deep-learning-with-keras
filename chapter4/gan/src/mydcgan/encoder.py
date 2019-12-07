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
import csv


class Convert():
    def __init__(self):

        self.train_size = 100000
        self.shape = (64, 64, 3)  # 画像サイズ
        self.z_dim = 100          # 潜在変数の次元

        # 学習済みのgenerator
        self.generator = self.generator_model()
        self.generator.load_weights(
            os.path.dirname(
                os.path.abspath(__file__)
            ) + "\\model\\generator"
        )
        self.generator.trainable = False

        # encoder
        self.encoder = self.encoder_model()

        self.combined = self.combined_model(self.generator, self.encoder)

        e_optim = Adam(lr=0.0002, beta_1=0.5)
        self.generator.compile(
            loss="binary_crossentropy",
            optimizer="Adam"
        )
        self.combined.compile(
            loss="mean_squared_error",
            optimizer=e_optim
        )
        self.encoder.compile(
            loss="mean_squared_error",
            optimizer="Adam"
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

    def encoder_model(self):
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
        model.add(Dense(100, activation="sigmoid"))

        model.summary()

        return model

    def combined_model(self, generator, encoder):
        model = Sequential()

        model.add(encoder)
        generator.trainable = False
        model.add(generator)

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

    def train(self, batch_size=32, n_epoch=100):

        for epoch in range(n_epoch):
            shuffle_idx = np.random.permutation(list(range(self.train_size)))

            print("Epoch is", epoch)
            print("Number of batches", int(self.train_size / batch_size))
            for index in range(int(self.train_size / batch_size)):

                image_batch = self.load_imgs(
                    shuffle_idx[index * batch_size:(index + 1) * batch_size]
                )

                image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5

                e_loss = self.combined.train_on_batch(
                    image_batch,
                    image_batch
                )

                print("batch %d e_loss : %f" % (index, e_loss))

            if epoch % 10 == 0:
                self.encoder.save_weights(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    ) + "\\model\\encoder_" + str(epoch),
                        True
                )

    def calc_att_vec(self, att_name, att_type, without=False, tol=100000):
        self.encoder.load_weights(
            os.path.dirname(
                os.path.abspath(__file__)
            ) + "\\model\\encoder_20"
        )

        att_vector = np.zeros((1, self.z_dim))
        num_img = 0

        print("calc vector with name:" + att_name + ", type:" + str(att_type))
        for i in range(0, self.train_size):
            if num_img > tol:
                break
            csv_path = os.path.dirname(
                os.path.abspath(__file__)
            ) + "\\attributes\\img_" + str(i) + ".csv"

            with open(csv_path) as f:
                for row in csv.reader(f):
                    name_in_csv = row[0].strip()
                    type_in_csv = int(row[1].strip())

                    if without:
                        if name_in_csv == att_name and type_in_csv != att_type:
                            img = self.load_imgs([i])
                            vec = self.encoder.predict(img, verbose=0)
                            num_img += 1

                            att_vector += vec
                            print(
                                "total:" + str(num_img) + ", " + "No. " + str(i))
                    else:
                        if name_in_csv == att_name and type_in_csv == att_type:
                            img = self.load_imgs([i])
                            vec = self.encoder.predict(img, verbose=0)
                            num_img += 1

                            att_vector += vec
                            print(
                                "total:" + str(num_img) + ", " + "No. " + str(i))

        att_vector /= num_img
        if without:
            output = "vec_" + att_name + "_without_" + str(att_type) + ".csv"
        else:
            output = "vec_" + att_name + "_" + str(att_type) + ".csv"

        np.savetxt(
            os.path.dirname(
                os.path.abspath(__file__)
            ) + "\\vector\\" + output,
            att_vector,
            delimiter=','
        )

    def gen_images(self):
        vec1_path = os.path.dirname(
            os.path.abspath(__file__)
        ) + "\\vector\\vec_glasses_10.csv"

        vec2_path = os.path.dirname(
            os.path.abspath(__file__)
        ) + "\\vector\\vec_glasses_11.csv"

        vec1 = np.loadtxt(vec1_path, delimiter=',')
        vec2 = np.loadtxt(vec2_path, delimiter=',')

        print(vec1.shape)
        # vec = np.reshape(vec1, (1, 100))

        np.random.seed(42)
        vec = np.random.uniform(-1, 1, (1, self.z_dim))
        gen_imgs_base = self.generator.predict(vec, verbose=0)
        gen_imgs_base = 0.5 * gen_imgs_base + 0.5
        fig, axs = plt.subplots(1, 1)
        axs.imshow(gen_imgs_base[0, :, :, :])
        axs.axis('off')
        path = os.path.dirname(
            os.path.abspath(__file__)
        ) + "\\vector\\imgs_base.png"
        fig.savefig(path)

        vec = np.reshape(vec1 - vec2, (1, 100)) + vec
        gen_imgs_glass_avg = self.generator.predict(vec, verbose=0)
        gen_imgs_glass_avg = 0.5 * gen_imgs_glass_avg + 0.5
        fig, axs = plt.subplots(1, 1)
        axs.imshow(gen_imgs_glass_avg[0, :, :, :])
        axs.axis('off')
        path = os.path.dirname(
            os.path.abspath(__file__)
        ) + "\\vector\\imgs_glass_avg.png"
        fig.savefig(path)

        plt.close()


if __name__ == "__main__":
    convert = Convert()

    # convert.train(
    #    batch_size=512,
    #    n_epoch=50s
    # )

    convert.calc_att_vec("glasses", 10, without=False, tol=100000)

    convert.gen_images()
