from dotenv import load_dotenv
import numpy as np
import pandas as pd
import argparse
import json

from epillid.metrics import mapk, average_precision_score, global_average_precision


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True,
                        help='The csv file of prediction scores for consumer vs reference images')
    parser.add_argument('--output', type=str, required=True,
                        help='The text file to generate')
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = parse_arguments()

    prediction_df = pd.read_csv(args.source)
    pred = prediction_df[prediction_df.columns[2:]].to_numpy()
    actual = np.zeros(pred.shape)
    actual[:, prediction_df['correct_label']] = 1

    n_sample, n_class = pred.shape

    actual_index = [np.where(r == 1)[0] for r in actual]
    pred_index = np.argsort(-pred, axis=1)

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

    with open(args.output, 'w+') as out:
        json.dump({
            'map_at_all': map_at_all,
            'map_at_1': map_at_1,
            'ap_sample': ap_sample,
            'ap_micro': ap_micro,
            'gap1': gap1,
            'gap2': gap2,
            'gap': gap,
        }, out)
