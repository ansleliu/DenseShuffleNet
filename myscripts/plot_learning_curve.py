######################################################################################################
# Title           :plot_learning_curve.py
# Description     :This script generates learning curves for caffe models
# usage           :python plot_learning_curve.py model_train.log ./caffe_model_learning_curve.png
######################################################################################################
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt


# Parsing training/validation logs
def parsing_logs(caffe_path, model_log_path, is_test):
    """
    Generating training and test logs
    """
    command = caffe_path + 'tools/extra/parse_log.sh ' + model_log_path
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

    # Read training and test logs
    train_log_path = model_log_path + '.train'
    test_log_path = model_log_path + '.test'
    train_log = pd.read_csv(train_log_path, delim_whitespace=True)
    if is_test == 1:
        test_log = pd.read_csv(test_log_path, delim_whitespace=True)
    else:
        test_log = None

    return train_log, test_log, train_log_path, test_log_path


def plot_curve(train_log, test_log, is_test, curve_path):
    """
    Making learning curve
    """
    fig, ax1 = plt.subplots()

    # Plotting training and test losses
    train_loss, = ax1.plot(train_log['#Iters'], train_log['TrainingLoss'], color='red',  alpha=.8)
    if is_test == 1:
        test_loss, = ax1.plot(test_log['#Iters'], test_log['TestLoss'], linewidth=1.6, color='green', marker="o", alpha=.8)
    ax1.set_ylim(ymin=0, ymax=10)
    ax1.set_yticks(np.arange(0, 20, 1.0))
    ax1.set_xlabel('Iterations', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=15)
    ax1.tick_params(labelsize=15)

    # Adding legend
    plt.legend([train_loss], ['Training Loss'], bbox_to_anchor=(1, 0.6))

    # plt.minorticks_on()
    plt.grid(b=True, which="major", linestyle="-.")
    plt.grid(b=True, which="minor", linestyle="-.")

    if is_test == 1:
        # Plotting test accuracy
        ax2 = ax1.twinx()
        # train_accuracy, = ax2.plot(train_log['#Iters'], test_log['TrainingAccuracy'], color='orange', alpha=.5)
        test_accuracy, = ax2.plot(test_log['#Iters'], test_log['TestAccuracy'],
                                  linewidth=1.6, color='blue', marker="o", alpha=.8)
        ax2.set_ylim(ymin=0, ymax=1.3)
        ax1.set_yticks(np.arange(0, 1.3, 0.1))
        ax2.set_ylabel('Accuracy', fontsize=15)
        ax2.tick_params(labelsize=15)

        # Adding legend
        plt.legend([train_loss, test_loss, test_accuracy],
                   ['Training Loss', 'Validation Loss', 'Validation Accuracy'],
                   bbox_to_anchor=(1, 0.6))

    plt.title('Training&Validation Curve', fontsize=18)

    # Saving learning curve
    plt.savefig(curve_path)


def delete_logs(train_log_path, test_log_path, is_test):
    """
    Deleting training and test logs
    """
    command = 'rm ' + train_log_path
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

    if is_test == 0:
        command = 'rm ' + test_log_path
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()

if __name__ == "__main__":
    # plt.style.use('ggplot')

    caffe_path = sys.argv[1]
    model_log_path = sys.argv[2]
    learning_curve_path = sys.argv[3]
    is_test = int(sys.argv[4])

    # Get directory where the model logs is saved, and move to it
    model_log_dir_path = os.path.dirname(model_log_path)
    os.chdir(model_log_dir_path)

    train_log, test_log, train_log_path, test_log_path = parsing_logs(caffe_path, model_log_path, is_test)
    plot_curve(train_log, test_log, is_test, learning_curve_path)
    delete_logs(train_log_path, test_log_path, is_test)
