"""apply dropout

"""

import numpy as np
from tensorflow_core.python.keras.api._v2.keras.datasets import mnist
from tensorflow_core.python.keras.api._v2.keras.models import Sequential
from tensorflow_core.python.keras.api._v2.keras.layers import Dense, Dropout, Activation
from tensorflow_core.python.keras.api._v2.keras.optimizers import SGD
from tensorflow_core.python.keras.api._v2.keras.utils import to_categorical
from make_tensorboard import make_tensorboard

# 乱数の固定
np.random.seed(1671)

# パラメータ設定
NB_EPOCH = 250          # エポック数
BATCH_SIZE = 128        # バッチサイズ
VERBOSE = 1             #
NB_CLASSES = 10         # 出力数
OPTIMIZER = SGD()       # 最適化関数
N_HIDDEN = 128          # 隠れ層数
VALIDATION_SPLIT = 0.2  # 検証データの割合
DROPOUT = 0.3  # ← 追加

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
# 層を定義
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))  # ← 784 × 128
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))  # ← 追加
model.add(Dense(N_HIDDEN))                           # ← 128 × 128
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))  # ← 追加
model.add(Dense(NB_CLASSES))                         # ← 128 × 10
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

callbacks = [make_tensorboard(set_dir_name='keras_MINST_V3')]

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
score = model.evaluate(X_test, Y_test, verbose=0)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
