import os
import glob
import pickle
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as np_utils
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=2, help='GPU device ID to run the job.')
    parser.add_argument('--data_dir', default=os.path.join('..', 'data', 'eeg_transfer_learning'), type=str)
    parser.add_argument('--model_dir', default=os.path.join('..', 'ckpt'), type=str)
    parser.add_argument('--result_dir', default=os.path.join('..', 'result'), type=str)
    return parser.parse_args()


def barplot(result_dir):
    def summary(fname):
        """
        Read a CSV file and create a spreadsheet-style pivot table as a DataFrame.
        :param fname: str
            File name of the saved result.
        :return: DataFrame
            An Excel style pivot table.
        """
        print("\n", fname.split(os.path.sep)[-1][:-4])
        dataframe = pd.read_csv(fname)
        print(dataframe.pivot_table(index=['Subject'], values=["Accuracy"], aggfunc=["mean", "std"], margins=True))
        return dataframe

    baseline_df = summary(os.path.join(result_dir, 'baseline.csv'))
    baseline_df['Is_transfer'] = "Baseline"
    baseline_df['Accuracy'] = baseline_df['Accuracy'].values * 100
    transfer_df = summary(os.path.join(result_dir, 'finetuning.csv'))
    transfer_df['Is_transfer'] = "Transfer learning"
    transfer_df['Accuracy'] = transfer_df['Accuracy'].values * 100
    score_df = pd.concat([baseline_df, transfer_df]).reset_index(drop=True)

    plt.figure(figsize=(25, 12))
    sns.barplot(data=score_df, x='Subject', y="Accuracy", hue='Is_transfer', palette='Blues',
                errwidth=1, capsize=.1)
    plt.legend(loc='upper right', ncol=3, frameon=False, fontsize=35, bbox_to_anchor=(1, 1.02))
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel('Subject', fontsize=40)
    plt.ylabel('Accuracy (%)', fontsize=40)
    plt.tight_layout()
    # plt.savefig(os.path.join(result_dir, 'barplot.png'), dpi=300)
    plt.show()

def plot_confusion_matrix(data_dir, model_dir, result_dir, type='baseline'):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    :param data_dir:
    :param model_dir:
    :param type: 'baseline' | 'fintuning'
    """
    def load_data(data_dir):
        with open(data_dir, 'rb') as f:
            data = pickle.load(f)
            test_x, test_y = data['test_x'], data['test_y']
            test_x = test_x[..., tf.newaxis].astype(np.float32)
            test_y = np_utils.to_categorical(test_y - np.min(test_y))
        return test_x, test_y

    def prediction(data_dir, model_dir):
        matrix = np.zeros((4,4))
        sub_data_list = sorted(glob.glob(os.path.join(data_dir, '*target.pkl')))
        sub_model_list = sorted(glob.glob(os.path.join(model_dir, '*')))
        for idx, sub_dir in enumerate(sub_model_list):
            x, y = load_data(sub_data_list[idx])
            fold_dir_list = sorted(glob.glob(os.path.join(sub_dir, type, '*')))
            for fold_dir in fold_dir_list:
                model = tf.keras.models.load_model(fold_dir)
                pred = model.predict(x)
                cm = confusion_matrix(y_true=y.argmax(axis=1), y_pred=pred.argmax(axis=1))
                matrix = matrix + cm
            matrix = matrix / len(fold_dir_list)
        matrix = matrix / len(sub_model_list)
        return matrix

    plt.figure(figsize=(9,7))
    sns.set(font_scale=1.4)
    target_names = ["Left hand", "Right hand", "Both feet", "Tongue"]
    matrix = prediction(data_dir, model_dir)
    normalized_cm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(normalized_cm, annot=True, xticklabels=target_names, yticklabels=target_names,
                cmap="GnBu", fmt='.2f', vmin=0, vmax=1, annot_kws={"size": 16})
    plt.ylabel('Actual', fontsize=16)
    plt.xlabel('Predicted', fontsize=16)
    plt.savefig(os.path.join(result_dir, '{}_cm.png'.format(type)), dpi=300)
    plt.show(block=False)

def main():
    args = get_args()
    print('Using GPU' + str(args.device_id))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    barplot(result_dir=args.result_dir)
    plot_confusion_matrix(data_dir=args.data_dir, model_dir=args.model_dir, result_dir=args.result_dir, type='baseline')
    plot_confusion_matrix(data_dir=args.data_dir, model_dir=args.model_dir, result_dir=args.result_dir, type='finetuning')

if __name__ == '__main__':
    main()
