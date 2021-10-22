import os
import numpy as np
import tensorflow as tf
from model.EEGNet import EEGNet
from utils.utils import random_seed, make_dir
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import StratifiedKFold


class Trainer(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.learning_rate = 0.001
        self.patience = 50
        self.cv_folders = 10

    @staticmethod
    def process_data(x, y):
        input_x = x[..., tf.newaxis].astype(np.float32)
        output_y = np_utils.to_categorical(y - np.min(y))
        return input_x, output_y

    def model(self):
        random_seed()
        model = EEGNet(nb_classes=4)
        return model

    def train(self, train_x, train_y, val_x, val_y, save_dir):
        train_x, train_y = self.process_data(train_x, train_y)
        val_x, val_y = self.process_data(val_x, val_y)

        random_seed()
        model = self.model()
        reset_model = tf.keras.models.clone_model(model)
        reset_model.set_weights(model.get_weights())
        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        reset_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics='CategoricalAccuracy')

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir,
                                                                 monitor='val_categorical_accuracy',
                                                                 mode='max',
                                                                 save_best_only=True)
        earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                                                  patience=self.patience)

        reset_model.fit(train_x, train_y, epochs=500, batch_size=64,
                        validation_data=(val_x, val_y),
                        callbacks=[checkpoint_callback, earlystopping_callback])

    def predict(self, test_x, test_y, load_dir):
        model = tf.keras.models.load_model(load_dir)
        acc = model.evaluate(test_x, test_y)[-1]
        return acc

    def cross_validation(self, train_x, train_y, **test_data):
        cv_list, acc_list = [], []
        for cv_idx, (train_index, val_index) in enumerate(
                StratifiedKFold(n_splits=self.cv_folders, random_state=117, shuffle=True).split(train_x, train_y)):

            # Set the detailed folder name for saving the cross validation models
            save_cv_dir = os.path.join(self.save_dir, f'{cv_idx + 1}fold')
            make_dir(save_cv_dir)

            cv_train_x, cv_train_y = train_x[train_index], train_y[train_index]
            cv_val_x, cv_val_y = train_x[val_index], train_y[val_index]

            self.train(train_x=cv_train_x, train_y=cv_train_y,
                       val_x=cv_val_x, val_y=cv_val_y, save_dir=save_cv_dir)

            if bool(test_data):
                test_x, test_y = test_data['test_x'], test_data['test_y']
                test_x, test_y = self.process_data(test_x, test_y)
            else:
                test_x, test_y = cv_val_x, cv_val_y
            acc = self.predict(test_x, test_y, load_dir=save_cv_dir)

            cv_list.append(cv_idx)
            acc_list.append(acc)

        return cv_list, acc_list
