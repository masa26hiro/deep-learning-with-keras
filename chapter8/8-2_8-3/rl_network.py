import os
import sys
from collections import deque
#from keras.models import Sequential
#from keras.layers.core import Dense, Flatten
#from keras.layers.convolutional import Conv2D
#from keras.optimizers import Adam
#from keras.models import clone_model
#from keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from PIL import Image
import numpy as np
import gym
import gym_ple  # noqa

class Agent(object):
    # 入力する画像サイズ（縦*横*フレーム）
    INPUT_SHAPE = (80, 80, 4)
    # 初期化
    def __init__(self, num_actions):
        # 行動の数
        ## 今回は3で固定だが、他のゲームにも対応できるような作りになっている
        self.num_actions = num_actions
        # モデルの定義
        ## 3つの畳み込み層->2つの全結合層
        ## activationとしてはreluを使用
        model = Sequential()
        model.add(Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=self.INPUT_SHAPE, kernel_initializer="normal",
            activation="relu"))
        model.add(Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer="normal",
            activation="relu"))
        model.add(Conv2D(
            64, kernel_size=3, strides=1, padding="same",
            kernel_initializer="normal",
            activation="relu"))
        model.add(Flatten())
        model.add(Dense(512, kernel_initializer="normal", activation="relu"))
        model.add(Dense(num_actions, kernel_initializer="normal"))
        self.model = model
    # 評価
    def evaluate(self, state, model=None):
        _model = model if model else self.model
        _state = np.expand_dims(state, axis=0)  # add batch size dimension
        # 実際にモデルで評価値を予測させる
        return _model.predict(_state)[0]
    # 行動
    def act(self, state, epsilon=0):
        # epsilon-greedy法
        ## epsilonより小さいならランダムに行動する
        if np.random.rand() <= epsilon:
            a = np.random.randint(low=0, high=self.num_actions, size=1)[0]
        else:
            q = self.evaluate(state)
            # 評価値が最大の行動を取得
            a = np.argmax(q)
        return a

# 観測
class Observer(object):

    def __init__(self, input_shape):
        self.size = input_shape[:2]  # width x height
        self.num_frames = input_shape[2]  # number of frames
        self._frames = []

    def observe(self, state):
        # グレースケール化
        g_state = Image.fromarray(state).convert("L")  # to gray scale
        # 想定しているsize(今回は80*80)に加工
        g_state = g_state.resize(self.size)  # resize game screen to input size
        g_state = np.array(g_state).astype("float")
        g_state /= 255  # scale to 0~1
        # もし最初の画面だった場合は4つに複製させる
        if len(self._frames) == 0:
            # full fill the frame cache
            self._frames = [g_state] * self.num_frames
        else:
            # 最新の画面を追加し、古い画面を出す処理
            self._frames.append(g_state)
            self._frames.pop(0)  # remove most old state

        input_state = np.array(self._frames)
        # change frame_num x width x height => width x height x frame_num
        input_state = np.transpose(input_state, (1, 2, 0))
        return input_state


# エージェントを学習させる処理
class Trainer(object):

    def __init__(self, env, agent, optimizer, model_dir=""):
        self.env = env
        self.agent = agent
        # Experience Replayに必要
        # 経験を蓄積させていく
        self.experience = []
        # 一定期間重みが固定されたモデルでQ値を出力したいので利用する
        self._target_model = clone_model(self.agent.model)
        self.observer = Observer(agent.INPUT_SHAPE)
        self.model_dir = model_dir
        if not self.model_dir:
            #self.model_dir = os.path.join(os.path.dirname(__file__), "model")
            self.model_dir = os.path.join(os.path.abspath("."), "model")
            if not os.path.isdir(self.model_dir):
                os.mkdir(self.model_dir)
        # mseで最適化
        self.agent.model.compile(optimizer=optimizer, loss="mse")
        self.callback = TensorBoard(self.model_dir)
        self.callback.set_model(self.agent.model)
    
    # バッチの取得をする
    def get_batch(self, batch_size, gamma):
        # self.experienceからランダムにバッチ数だけデータを取得する
        batch_indices = np.random.randint(
            low=0, high=len(self.experience), size=batch_size)
        X = np.zeros((batch_size,) + self.agent.INPUT_SHAPE)
        y = np.zeros((batch_size, self.agent.num_actions))
        for i, b_i in enumerate(batch_indices):
            s, a, r, next_s, game_over = self.experience[b_i]
            X[i] = s
            y[i] = self.agent.evaluate(s)
            # future reward
            # 次の時点での将来的に得られる最大値を取得
            # 一定時間重みを固定したモデルを計算に利用する
            Q_sa = np.max(self.agent.evaluate(next_s,
                                              model=self._target_model))
            # ベルマン方程式
            # Q(s_t, a_t) = r + gamma*Q(s_t+1, a_t+1)
            if game_over:
                y[i, a] = r
            else:
                y[i, a] = r + gamma * Q_sa
        return X, y

    def write_log(self, index, loss, score):
        for name, value in zip(("loss", "score"), (loss, score)):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.callback.writer.add_summary(summary, index)
            self.callback.writer.flush()
    
    def train(self,
              gamma=0.99,
              initial_epsilon=0.1, final_epsilon=0.0001,
              memory_size=50000,
              observation_epochs=100, training_epochs=2000,
              batch_size=32, render=True):
        self.experience = deque(maxlen=memory_size)
        # 最初に100回観測したあと、2000回学習させるイメージ
        epochs = observation_epochs + training_epochs
        epsilon = initial_epsilon
        model_path = os.path.join(self.model_dir, "agent_network.h5")
        fmt = "Epoch {:04d}/{:d} | Loss {:.5f} | Score: {} | e={:.4f} train={}"
        #学習ループ
        for e in range(epochs):
            loss = 0.0
            rewards = []
            # 環境を初期化させる（スタート画面に戻すような感じ）
            initial_state = self.env.reset()
            state = self.observer.observe(initial_state)
            game_over = False
            is_training = True if e > observation_epochs else False

            # let's play the game
            while not game_over:
                if render:
                    self.env.render()
                # 学習してないときは完全にランダムに動かす
                if not is_training:
                    action = self.agent.act(state, epsilon=1)
                else:
                    action = self.agent.act(state, epsilon)
                # 次の状態、報酬、ゲームオーバーになったか、を取得
                next_state, reward, game_over, info = self.env.step(action)
                next_state = self.observer.observe(next_state)
                # 行動、報酬、次の状態といった一連の情報をexperienceに追加していく
                self.experience.append(
                    (state, action, reward, next_state, game_over)
                    )

                rewards.append(reward)
            
                if is_training:
                    # バッチ作成、学習
                    X, y = self.get_batch(batch_size, gamma)
                    loss += self.agent.model.train_on_batch(X, y)

                state = next_state

            loss = loss / len(rewards)
            score = sum(rewards)

            if is_training:
                self.write_log(e - observation_epochs, loss, score)
                # 重み更新
                self._target_model.set_weights(self.agent.model.get_weights())
            # epsilonを徐々に小さくしていく
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / epochs

            print(fmt.format(e + 1, epochs, loss, score, epsilon, is_training))

            if e % 100 == 0:
                self.agent.model.save(model_path, overwrite=True)

        self.agent.model.save(model_path, overwrite=True)

def main(render):
    env = gym.make("Catcher-v0")
    num_actions = env.action_space.n
    agent = Agent(num_actions)
    trainer = Trainer(env, agent, Adam(lr=1e-6))
    trainer.train(render=render)

if __name__ == "__main__":
    render = False if len(sys.argv) < 2 else True
    main(render)