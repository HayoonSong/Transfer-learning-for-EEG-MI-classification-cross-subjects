import os
import glob
import pickle
import argparse
from model.EEGNet import EEGNet
from Trainer import Trainer
from utils.utils import random_seed, make_dir
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=1, help='GPU device ID to run the job.')
    parser.add_argument('--data_dir', default=os.path.join('..', '..', 'data', ''), type=str)
    parser.add_argument('--ckpt_dir', default=os.path.join('..', '..', 'ckpt'), type=str)

    parser.add_argument('--nb_classes', default=4, type=int)
    parser.add_argument('--chans', default=22, type=int)
    parser.add_argument('--sfreq', default=250, type=int)
    parser.add_argument('--t_min', default=2.5, type=int)
    parser.add_argument('--t_max', default=4.5, type=int)
    return parser.parse_args()


class PretrainingTrainer(Trainer):
    def __init__(self, args, save_dir):
        super(PretrainingTrainer, self).__init__(save_dir=save_dir)
        self.args = args

    def model(self):
        args = self.args
        random_seed()
        model = EEGNet(nb_classes=args.nb_classes, Chans=args.chans,
                       Samples=int(args.sfreq * (args.t_max - args.t_min)),
                       kernLength=int(args.sfreq / 2))
        return model


def load_source_data(data_dir):
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
        x, y, = data['source_x'], data['source_y']
    return x, y


def subject_specific_model():
    args = get_args()

    # Set GPU device id
    print('Using GPU' + str(args.device_id))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

    data_dir = sorted(glob.glob(os.path.join(args.data_dir, 'eeg_transfer_learning', '*_source.pkl')))
    for sub_dir in data_dir:
        # Set the detailed folder name for saving the subject models
        sub_id = sub_dir.split(os.path.sep)[-1][:2]
        save_model_dir = os.path.join(args.ckpt_dir, 'sub_' + sub_id, 'pretraining')
        make_dir(save_model_dir)

        x, y = load_source_data(sub_dir)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=117)
        trainer = PretrainingTrainer(args, save_model_dir)
        trainer.train(train_x, train_y, test_x, test_y, save_dir=save_model_dir)
        test_x, test_y = trainer.process_data(test_x, test_y)
        acc = trainer.predict(test_x, test_y, save_model_dir)
        print('Subject {} pre-training accuracy: {}%'.format(sub_id, acc * 100))


if __name__ == '__main__':
    subject_specific_model()
    exit()
