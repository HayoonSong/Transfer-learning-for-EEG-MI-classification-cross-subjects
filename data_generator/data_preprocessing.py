import os
import pickle
import argparse
import glob
import numpy as np
from scipy import io
from utils.utils import make_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.join('..', '..', 'data', ''), type=str)
    parser.add_argument('--chans', default=22, type=int)
    parser.add_argument('--sfreq', default=250, type=int)
    parser.add_argument('--t_min', default=2.5, type=int)
    parser.add_argument('--t_max', default=4.5, type=int)
    return parser.parse_args()


class BCICompetitionIV2a(object):
    def __init__(self, arguments):
        self.args = arguments
        self.t_sess_path = sorted(glob.glob(self.args.data_dir + "*T.mat"))
        self.e_sess_path = sorted(glob.glob(self.args.data_dir + "*E.mat"))
        self.sub_list = sorted([sub.split(os.path.sep)[-1][1:3] for sub in self.t_sess_path])

    def process_data(self, data_dir):
        """
        :param data_dir: The raw data path (e.g., A01E.mat)
        :return: X, Y
        """
        args = self.args
        trials_len = 6 * 48
        time_point = int(args.sfreq * (args.t_max - args.t_min))

        trial_idx = 0
        x = np.zeros((trials_len, args.chans, time_point))
        y = np.zeros(trials_len)

        a = io.loadmat(data_dir)
        a_data = a['data']
        for ii in range(0, a_data.size):
            a_data1 = a_data[0, ii]
            a_data2 = [a_data1[0, 0]]
            a_data3 = a_data2[0]
            a_x = a_data3[0]
            a_trial = a_data3[1]
            a_y = a_data3[2]
            a_artifacts = a_data3[5]

            for trial in range(0, a_trial.size):  # trial = 0 (ex, EOG), pass
                if (a_artifacts[trial] == 0):
                    mi_start = int(a_trial[trial] + (args.sfreq * args.t_min))
                    mi_end = int(mi_start + time_point)

                    x[trial_idx, :, :] = np.transpose(a_x[mi_start:mi_end, :22])
                    y[trial_idx] = int(a_y[trial])
                    trial_idx += 1

        x = x[0:trial_idx, :, :]
        y = y[0:trial_idx]

        return x, y

    def load_target_data(self, sub):
        sub_idx = self.sub_list.index(sub)
        train_x, train_y = self.process_data(self.t_sess_path[sub_idx])
        test_x, test_y = self.process_data(self.e_sess_path[sub_idx])
        print("Target subject: {}".format(sub))
        print("Train_x: {}, Train_y: {}, Test_x: {}, Test_y: {}".format(
            train_x.shape, train_y.shape, test_x.shape, test_y.shape))
        return train_x, train_y, test_x, test_y

    # Concatenate the source subjects' data except the target subject
    def load_source_data(self, sub):
        sub_list = self.sub_list.copy()
        t_sess_path = self.t_sess_path.copy()
        e_sess_path = self.e_sess_path.copy()
        sub_idx = sub_list.index(sub)

        del sub_list[sub_idx]
        del t_sess_path[sub_idx]
        del e_sess_path[sub_idx]

        # Combine the training data and the evaluation data of a subject's eeg data
        total_x, total_y = [], []
        for train_dir, eval_dir in zip(t_sess_path, e_sess_path):
            t_x, t_y = self.process_data(train_dir)
            e_x, e_y = self.process_data(eval_dir)
            sub_x = np.concatenate((t_x, e_x))
            sub_y = np.concatenate((t_y, e_y))
            total_x.append(sub_x)
            total_y.append(sub_y)

        total_x = np.concatenate(total_x)
        total_y = np.concatenate(total_y)
        print("Source subjects: {}".format(sub_list))
        print("Source_x: {}, Source_y: {}".format(total_x.shape, total_y.shape))
        return total_x, total_y

    def load_data(self, sub):
        args = self.args

        # Set the detailed folder name for saving the data
        output_dir = os.path.join(args.data_dir, 'eeg_transfer_learning')
        make_dir(output_dir)

        train_x, train_y, test_x, test_y = self.load_target_data(sub)
        source_x, source_y = self.load_source_data(sub)

        # Save
        save_target_dict = {
            "train_x": train_x,
            "train_y": train_y,
            "test_x": test_x,
            "test_y": test_y
        }
        save_source_dict = {
            "source_x": source_x,
            "source_y": source_y}

        with open(os.path.join(output_dir, f'{sub}_target.pkl'), 'wb') as f:
            pickle.dump(save_target_dict, f)
        with open(os.path.join(output_dir, f'{sub}_source.pkl'), 'wb') as g:
            pickle.dump(save_source_dict, g)

        print("\n=======================================\n")


def main():
    args = get_args()
    data_helper = BCICompetitionIV2a(args)
    for sub in data_helper.sub_list:
        data_helper.load_data(sub)


if __name__ == '__main__':
    main()
