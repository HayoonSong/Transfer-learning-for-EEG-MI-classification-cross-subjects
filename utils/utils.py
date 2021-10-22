import os
import numpy as np
import tensorflow as tf
import random as rn


def random_seed(seed_num=117):
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    np.random.seed(seed_num)
    rn.seed(seed_num)
    tf.random.set_seed(seed_num)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
