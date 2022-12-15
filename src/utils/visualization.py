import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def plot_conf_matrix(cm, output_dir, class_names):
    df_cm = pd.DataFrame(cm, index=[label for label in class_names],
                         columns=[label for label in class_names])

    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm / np.sum(df_cm), annot=True, fmt='.4f', cmap='Blues')
    plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix.png'))
    plt.clf()

    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(output_dir, 'relative_confusion_matrix.png'))
    plt.clf()


def plot_clf_report(clf_report, output_dir):
    sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    plt.savefig(os.path.join(output_dir, "clf_report.png"))