import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from Trainer import Trainer
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.EEGNet import EEGNet
from utils import make_dir, random_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=0, help='GPU device ID to run the job.')
    parser.add_argument('--data_dir', default=os.path.join('..', '..', 'data'), type=str)
    parser.add_argument('--ckpt_dir', default=os.path.join('..', '..', 'ckpt'), type=str)
    parser.add_argument('--result_dir', default=os.path.join('..', '..', 'result'), type=str)

    parser.add_argument('--nb_classes', default=4, type=int)
    parser.add_argument('--chans', default=22, type=int)
    parser.add_argument('--sfreq', default=250, type=int)
    parser.add_argument('--t_min', default=2.5, type=int)
    parser.add_argument('--t_max', default=4.5, type=int)
    return parser.parse_args()


class BaselineTrainer(Trainer):
    def __init__(self, args, save_dir):
        super(BaselineTrainer, self).__init__(save_dir=save_dir)
        self.args = args

    def model(self):
        args = self.args
        random_seed()
        model = EEGNet(nb_classes=args.nb_classes, Chans=args.chans,
                       Samples=int(args.sfreq * (args.t_max - args.t_min)),
                       kernLength=int(args.sfreq / 2))
        return model


def load_target_data(data_dir):
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
        train_x, train_y, = data['train_x'], data['train_y']
        test_x, test_y = data['test_x'], data['test_y']
    return train_x, train_y, test_x, test_y


def save_result(sub_list, cv_list, acc_list, result_dir):
    df = pd.DataFrame({'Subject': sub_list, 'CV': cv_list, 'Accuracy': acc_list})
    df.reset_index(drop=True)
    df.to_csv(os.path.join(result_dir, 'baseline.csv'), index=False)


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
        save_model_dir = os.path.join(args.ckpt_dir, 'sub_' + sub_id, 'baseline')
        make_dir(save_model_dir)

        train_x, train_y, test_x, test_y = load_target_data(sub_dir)
        trainer = BaselineTrainer(args, save_model_dir)
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
