from dotenv import load_dotenv
import os
import numpy as np
from epillid.metrics import apk, mapk, average_precision_score, global_average_precision
import pandas as pd

from epillid_datasets import get_label_encoder

if __name__ == '__main__':
    load_dotenv()

    prediction_df = pd.read_csv('predicted.csv')

    label_encoder = get_label_encoder(os.getenv('EPILLID_DATASET_ROOT'))
    labels = [str(i) for i in range(len(label_encoder.classes_))]
    pred = prediction_df[labels].to_numpy()
    actual_df = prediction_df['correct_label']
    actual = np.zeros(pred.shape)
    actual[:, actual_df] = 1

    n_sample, n_class = pred.shape

    actual_index = [np.where(r==1)[0] for r in actual]
    pred_index = np.argsort(-pred, axis=1)
    # print(pred)
    # print(actual)
    # print(actual_index)
    # print(pred_index)

    # print('=' * 30)
    # for i in range(n_sample):
    #     print(apk(actual_index[i], pred_index[i], k=1), pred_index[i])
    #     print(apk(actual_index[i], pred_index[i], k=2), pred_index[i])
    #     print(apk(actual_index[i], pred_index[i]), pred_index[i])
    #     print('-' * 30)

    map_at_all = mapk(actual_index, pred_index, k=n_class)
    map_at_1 = mapk(actual_index, pred_index, k=1)
    print('map')
    print(map_at_all, map_at_1)

    ap_sample = average_precision_score(actual, pred, average="samples")
    ap_micro = average_precision_score(actual, pred, average="micro")
    print('ap sample, micro')
    print(ap_sample, ap_micro)

    gap1 = global_average_precision(actual, pred, k=1)
    gap2 = global_average_precision(actual, pred, k=2)
    gap = global_average_precision(actual, pred)

    print('gap')
    print(gap1, gap2, gap)

    # map and ap_samples should be same
    assert abs(ap_sample - map_at_all) < 0.01
    assert abs(ap_micro - gap) < 0.01
