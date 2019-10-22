import os
from time import gmtime, strftime
from tensorflow_core.python.keras.api._v2.keras.callbacks import TensorBoard


def make_tensorboard(set_dir_name=''):
    tictoc = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
    directory_name = tictoc
    log_dir = set_dir_name + '_' + directory_name
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir)
    return tensorboard
