import os
import argparse

import numpy as np
import ipdb
from collections import Counter


def get_evaluations(scores):
    e = []
    for i in range(len(scores) // 2):
        e.append([scores[2 * i], scores[2 * i + 1]])
    return np.array(e)


def main(params):
    pred_file = os.path.join(params.pred_dir, params.prediction_file)

    print(f'Reading prediction file {params.prediction_file} from {params.pred_dir}')

    with open(pred_file) as f:
        pred = [line.split() for line in f.read().split('\n')]

    all_triple_set = set()
    for set_type in ['train', 'valid', 'test']:
        tpls = [tuple(l.strip().split('\t')) for l in open(f'{params.data_dir}/{set_type}.txt')]
        all_triple_set = all_triple_set.union(tpls)

    # testing
    tuples = []
    h_lists = []
    t_lists = []
    for i in range(len(pred) // 3):
        tuples.append([pred[3 * i][0], pred[3 * i][1], pred[3 * i][2]])
        h_lists.append(get_evaluations(pred[3 * i + 1][1:]))
        t_lists.append(get_evaluations(pred[3 * i + 2][1:]))

    TOT_ENTITIES = len(tuples)

    head_ranks = []
    tail_ranks = []
    head_ranks_filtered = []
    tail_ranks_filtered = []
    for i, (tple, h_list, t_list) in enumerate(zip(tuples, h_lists, t_lists)):
        h_idx = [] if len(h_list) == 0 else np.argwhere(h_list[:, 0] == tple[0])
        t_idx = [] if len(t_list) == 0 else np.argwhere(t_list[:, 0] == tple[2])

        if len(h_idx) > 0:
            head_score = float(h_list[h_idx[0][0], 1])  # str

            head_rank = 0
            head_rank_filtered = 0
            for h, s_str in h_list:
                s = float(s_str)
                if s >= head_score:
                    head_rank += 1
                    if (h == tple[0]) or ((h, tple[1], tple[2]) not in all_triple_set):
                        head_rank_filtered += 1
                else:
                    break
            head_ranks.append(head_rank)
            head_ranks_filtered.append(head_rank_filtered)
        else:
            head_ranks.append(TOT_ENTITIES)
            head_ranks_filtered.append(TOT_ENTITIES)

        if len(t_idx) > 0:
            tail_score = float(t_list[t_idx[0][0], 1])
            tail_rank = 0
            tail_rank_filtered = 0
            for t, s_str in t_list:
                s = float(s_str)
                if s >= tail_score:
                    tail_rank += 1
                    if (t == tple[2]) or ((tple[0], tple[1], t) not in all_triple_set):
                        tail_rank_filtered += 1
                else:
                    break
            tail_ranks.append(tail_rank)
            tail_ranks_filtered.append(tail_rank_filtered)

        else:
            tail_ranks.append(TOT_ENTITIES)
            tail_ranks_filtered.append(TOT_ENTITIES)


    ranks = head_ranks + tail_ranks
    # for ranks in [head_ranks, tail_ranks]:
    hits = Counter()
    for hit in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]:
        hits[f'hits@{hit}'] = len([x for x in ranks if x <= hit])

    print('TOT_ENTITIES', TOT_ENTITIES)
    print('raw:')
    print({k: v for k, v in hits.items()})
    print({k: v/TOT_ENTITIES/2 for k, v in hits.items()})

    print('filtered:')
    ranks_filtered = head_ranks_filtered + tail_ranks_filtered
    hits = Counter()
    for hit in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]:
        hits[f'hits@{hit}'] = len([x for x in ranks_filtered if x <= hit])

    print({k: v for k, v in hits.items()})
    print({k: v/TOT_ENTITIES/2 for k, v in hits.items()})
    ipdb.set_trace()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='eval predictions')

    parser.add_argument("--dataset", "-d", type=str, default="tmp1",
                        help="Dataset string")
    parser.add_argument("--prediction_file", "-f", type=str, default="pos_predictions.txt",
                        help="Dataset string")

    params = parser.parse_args()
    params.data_dir = f'./data/{params.dataset}'
    params.pred_dir = './'
    # params.data_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '../data/') + params.dataset

    main(params)
