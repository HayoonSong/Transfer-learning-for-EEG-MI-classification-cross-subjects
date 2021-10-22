import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from trainer.Trainer import Trainer
from utils import make_dir
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.constraints import max_norm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=0, help='GPU device ID to run the job.')
    parser.add_argument('--data_dir', default=os.path.join('..', '..', 'data', ''), type=str)
    parser.add_argument('--ckpt_dir', default=os.path.join('..', '..', 'ckpt'), type=str)
    parser.add_argument('--result_dir', default=os.path.join('..', '..', 'result'), type=str)

    parser.add_argument('--nb_classes', default=4, type=int)
    return parser.parse_args()


class FinetuningTrainer(Trainer):
    def __init__(self, args, save_dir):
        super(FinetuningTrainer, self).__init__(save_dir=save_dir)
        self.args = args

    def model(self):
        pre_model_dir = os.path.join(self.save_dir, '..', 'pretraining')
        pre_model = tf.keras.models.load_model(pre_model_dir)
        extracted_layers = pre_model.layers[:-2]
        extracted_layers.append(Dense(self.args.nb_classes, name='dense', kernel_constraint=max_norm(0.25)))
        extracted_layers.append(Activation('softmax', name='softmax'))
        finetuning_model = tf.keras.Sequential(extracted_layers)
        return finetuning_model


def load_target_data(data_dir):
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
        train_x, train_y, = data['train_x'], data['train_y']
        test_x, test_y = data['test_x'], data['test_y']
    return train_x, train_y, test_x, test_y


def save_result(sub_list, cv_list, acc_list, result_dir):
    df = pd.DataFrame({'Subject': sub_list, 'CV': cv_list, 'Accuracy': acc_list})
    df.reset_index(drop=True)
    df.to_csv(os.path.join(result_dir, 'finetuning.csv'), index=False)


def subject_specific_model():
    args = get_args()

    # Set the detailed folder name for saving the result
    make_dir(args.result_dir)

    # Set GPU device id
    print('Using GPU' + str(args.device_id))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

    data_dir = sorted(glob.glob(os.path.join(args.data_dir, 'eeg_transfer_learning', '*_target.pkl')))
    sub_list, cv_list, acc_list = [], [], []
    for sub_dir in data_dir:

        # Set the detailed folder name for saving the subject models
        sub_id = sub_dir.split(os.path.sep)[-1][:2]
        save_model_dir = os.path.join(args.ckpt_dir, 'sub_'+sub_id, 'finetuning')
        make_dir(save_model_dir)

        train_x, train_y, test_x, test_y = load_target_data(sub_dir)
        trainer = FinetuningTrainer(args, save_model_dir)
        cv, acc = trainer.cross_validation(train_x, train_y, test_x=test_x, test_y=test_y)
        print("Subject {} accuracy: {}%".format(sub_id, np.mean(acc) * 100))

        sub_list.append([sub_id] * trainer.cv_folders)
        cv_list.append(cv)
        acc_list.append(acc)

    sub_list = np.concatenate(sub_list)
    cv_list = np.concatenate(cv_list)
    acc_list = np.concatenate(acc_list)
    save_result(sub_list, cv_list, acc_list, result_dir=args.result_dir)


if __name__ == '__main__':
    subject_specific_model()
    exit()
