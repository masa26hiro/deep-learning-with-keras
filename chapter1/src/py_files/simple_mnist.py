"""simple mnist

"""

import numpy as np
from tensorflow_core.python.keras.api._v2.keras.datasets import mnist
from tensorflow_core.python.keras.api._v2.keras.models import Sequential
from tensorflow_core.python.keras.api._v2.keras.layers import Dense, Activation
from tensorflow_core.python.keras.api._v2.keras.optimizers import SGD
from tensorflow_core.python.keras.api._v2.keras.utils import to_categorical
from .make_tensorboard import make_tensorboard

# 乱数の固定
np.random.seed(42)

# パラメータ設定
NB_EPOCH = 200          # エポック数
BATCH_SIZE = 128        # バッチサイズ
VERBOSE = 1             #
NB_CLASSES = 10         # 出力数
OPTIMIZER = SGD()       # 最適化関数
N_HIDDEN = 128          # 隠れ層数
VALIDATION_SPLIT = 0.2  # 検証データの割合

# データの読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()
RESHAPED = 784  # 28×28 = 784

# データの整形
TRAIN_DATA_SIZE = 60000
TEST_DATA_SIZE = 10000
X_train = X_train.reshape(TRAIN_DATA_SIZE, RESHAPED)
X_test = X_test.reshape(TEST_DATA_SIZE, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 正規化
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train sample')
print(X_test.shape[0], 'test sample')

# ラベルをOne-hotエンコーディング
Y_train = to_categorical(y_train, NB_CLASSES)
Y_test = to_categorical(y_test, NB_CLASSES)

# Sequentialモデルを定義
model = Sequential()
# 層を追加　入力 784, 出力10, 密結合, ソフトマックス関数
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
# モデルを評価しTensorBoard互換のログに書き込む
model.summary()

# モデルをコンパイル
# -損失関数　：カテゴリカルクロスエントロピー
# -最適化関数：確率的勾配降下法
# -評価関数　：Accuracy(正解率)
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

callbacks = [make_tensorboard(set_dir_name='keras_MINST_V1')]

# 学習
# -バッチサイズ　　：128
# -エポック数　　　：200
# -コールバック関数：make_tensorboard(訓練の各段階で呼び出される関数)
# -ＶＥＲＢＯＳＥ　：詳細表示モードON
# -検証データの割合：20%
model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE,
          epochs=NB_EPOCH,
          callbacks=callbacks,
          verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT)

# スコアを表示
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
