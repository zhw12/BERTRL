import ipdb
import numpy as np
import torch
from sklearn import metrics
from collections import defaultdict, Counter
import argparse
from sklearn.metrics import average_precision_score
import itertools

# from transformers import AutoTokenizer
import re
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--bertrl_output_dir", "-o", type=str,
                        help="Output", default='')
    parser.add_argument("--additional_suffix", "-suf", type=str,
                        help="Output", default='')
    parser.add_argument("--show_rel", default=False, action='store_true',
                        help="show_rel ")
    parser.add_argument('--head_tail_type', '-ht', type=str, default='',
                        help='head_tail_type')
    parser.add_argument("--total_triples", type=int, default=0,
                        help="total_triples")

    params = parser.parse_args()
    params.main_dir = '/scratch/home/hanwen/BERTRL'

    # an_sufix = '_anonymized' if params.anonymized else ''
    # data_dir = '/scratch/home/hanwen/BertLogic/output_WN18RR_v1_ind/'
    # data_dir = f'{params.main_dir}/output_{params.dataset}_ind{an_sufix}'

    if not params.bert_output_dir:
        params.bert_output_dir = f'{params.main_dir}/output_{params.dataset}{params.additional_suffix}'

    res = np.load(f'{params.bert_output_dir}/test_results_prediction_scores.npy')
    t_res = torch.softmax(torch.from_numpy(res), 1)

    bertrl_data_dir = f'{params.main_dir}/bertrl_data/{params.dataset}'
    bertrl_data_file = f'{bertrl_data_dir}/test.tsv'

    
    examples = [l.strip().split('\t') for l in open(bertrl_data_file)][1:]

    assert len(examples) == res.shape[0], "score should have same size with examples"

    dataset2triples = {'WN18RR_ind': 188*2, 'fb237_ind': 205*2, 'nell_ind': 476*2, 'fb237':492*2, 'WN18RR':638*2, 'nell':968*2,
    'fb237_ind': 478*2, 'WN18RR_ind':441*2, 'nell_ind':100*2}

    data_dir = f'{params.main_dir}/data'
    for dataset_dir in os.listdir(data_dir):
        test_file = os.path.join(data_dir, dataset_dir, 'test.txt')
        if os.path.exists(test_file):
            dataset2triples[dataset_dir] = len([l for l in open(test_file)]) * 2

    # ipdb.set_trace()

    dataset_short = '_'.join(params.dataset.split('_')[:2])

    triple_test_data_dir = f'{params.main_dir}/data/{dataset_short}'
    if 'inductive' in params.dataset:
        triple_test_data_dir += '_ind'
    triple_test_file = f'{triple_test_data_dir}/test.txt'
    triples_test = [l.strip().split('\t') for l in open(triple_test_file)]
    
    part = ''
    if re.search('_rel\d+', params.dataset):
        part = re.search('_rel\d+', params.dataset)[0]

    # part = '_rel50'

    triple_train_file = f'{params.main_dir}/data/{dataset_short}/train{part}.txt'
    triples_train = [l.strip().split('\t') for l in open(triple_train_file)]
    train_rels = set([tpl[1] for tpl in triples_train])
    test_rels = set([tpl[1] for tpl in triples_test])


    eid2relation = {i:tpl[1] for i, tpl in enumerate(triples_test * 2)}

    if 'inductive' in params.dataset:
        total_triples = dataset2triples.get(dataset_short+'_ind', 1000000)
    else:
        total_triples = dataset2triples.get(dataset_short, 1000000)

    eid2pos = {}
    eid2neg = defaultdict(list)
    labels = []

    eid2wrongs = defaultdict(list)
    eid2corrects = defaultdict(list)

    # another set of evaluation
    eid2pos = defaultdict(list)
    eid2neg = defaultdict(list)
    eid2neg_lv2 = {}

    for i, example in enumerate(examples):
        label = int(example[0])
        eid = example[1].split('-')[2]  # test-neg-2-24, train-pos-2
        eid = int(eid)

        # if label == 1:
        if 'pos' in example[1]:
            assert label == 1
            eid2pos[eid].append(i)
        else:
            e_negid = example[1].split('-')[3]  # test-neg-2-24
            eid2neg[eid].append(i)
            if eid not in eid2neg_lv2:
                eid2neg_lv2[eid] = defaultdict(list)
            eid2neg_lv2[eid][e_negid].append(i)

    if params.head_tail_type:
        total_triples //= 2

    if params.total_triples:
        total_triples = total_triples

    rel2testcnt = Counter()
    for tpl in triples_test * 2:
        rel2testcnt[tpl[1]] += 1

    hit1 = 0
    hits = Counter()
    rel2hits = defaultdict(Counter)
    rel2hits_base = defaultdict(Counter)
    ranks = []
    for eid, pos_is in eid2pos.items(): # all pos eids with examples

        # if params.head_tail_type == 'tail':
        #     if eid >= total_triples:
        #         continue
        # elif params.head_tail_type == 'head':
        #     if eid < total_triples:
        #         continue    
        # else:
        #     pass    

        pos_scores = t_res[pos_is, 1]
        pos_max_score = torch.max(pos_scores).item()

        neg_is = eid2neg[eid]
        neg_eids = eid2neg_lv2.get(eid, [])
        geq_j = 1  # rank
        if not neg_is:  # no negative, only positive
            hit1 += 1
            eid2corrects[eid].append([pos_is, pos_scores, [], []])
        else:
            neg_scores = t_res[neg_is, 1]
            neg_max_score = torch.max(neg_scores).item()

            neg_scores_of_eid = []
            neg_lv2_scores_of_eid = defaultdict(list)
            neg_scores_lists = []
            for neg_eid in eid2neg_lv2[eid]:
                neg_is_of_eid = eid2neg_lv2[eid][neg_eid]
                neg_scores_ = t_res[neg_is_of_eid, 1]  # previously a bug here, previously use neg_is which lowers the hit 2+ performance
                neg_max_score_ = torch.max(neg_scores_).item()
                neg_scores_of_eid.append(neg_max_score_)
                neg_scores_lists.append(neg_scores_.sort(0, descending=True).values.tolist())

            _scores_pos = pos_scores.sort(0, descending=True).values.tolist()
            for _scores in neg_scores_lists:
                for s1, s2 in itertools.zip_longest(_scores_pos, _scores, fillvalue=100):
                    if s1 < s2 or s1 == 100:
                        geq_j += 1
                        break
                    elif s1 == s2:
                        continue
                    else:
                        break
            
            if geq_j <= 1:
                eid2corrects[eid].append([pos_is, pos_scores, neg_is, neg_scores])
            else:
                if eid not in eid2wrongs:
                    eid2wrongs[eid].append([pos_is, pos_scores, neg_is, neg_scores])

            # for neg_score in neg_scores_of_eid:
            #     if pos_max_score <= neg_score:
            #         geq_j += 1

            # if pos_max_score > neg_max_score:
            #     hit1 += 1
            #     eid2corrects[eid].append([pos_is, pos_scores, neg_is, neg_scores])
            # else:
            #     if eid not in eid2wrongs:
            #         eid2wrongs[eid].append([pos_is, pos_scores, neg_is, neg_scores])
        
        cnt_neg_of_eid = len(eid2neg_lv2.get(eid, []))
        rel = eid2relation[eid]
        for hit in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]:
            if geq_j <= hit:
                hits[hit] += 1
                rel2hits[rel][hit] += 1

            if cnt_neg_of_eid + 1 <= hit:
                rel2hits_base[rel][hit] += 1

        ranks.append(geq_j)

    if len(ranks) < total_triples:
        ranks += [50] * (total_triples-len(ranks))

    hits = {k: hits[k] for k in sorted(hits)}

    mrr = np.mean(1 / np.array(ranks))
    print('mrr:', mrr)

    print(hits)
    print({k:round(v/total_triples, 3) for k,v in hits.items()})

    def show(eid, item=None, k=5):
        if item is None:
            if eid in eid2wrongs:
                item = eid2wrongs[eid]
            else:
                item = eid2corrects[eid]
        # for eid, item in eid2wrongs.items():
        pos_is, pos_scores, neg_is, neg_scores = item[0]
        pos_max_i = pos_is[torch.argmax(pos_scores)]  # use pos_is to reconstruct example i
        neg_max_i = neg_is[torch.argmax(neg_scores)]

        neg_topk_is = [neg_is[ind] for ind in torch.topk(neg_scores, k).indices.tolist()]
        neg_topk_scores = torch.topk(neg_scores, 5).values

        pos_max_score = torch.max(pos_scores).item()
        neg_max_score = torch.max(neg_scores).item()

        # print(pos_max_score, neg_max_score)
        # print(examples[pos_max_i], examples[neg_max_i], sep='\n')
        print(examples[pos_max_i])

        print('eid:', eid)
        print('max pos score:', pos_max_score, 
            'max neg score:', neg_max_score)
        print('-----')
        print('max pos i:', pos_max_i, 'max neg i:', neg_max_i)
        for i in pos_is:
            print(i, t_res[i, 1].item(), '\t', examples[i])



        for i in neg_topk_is:
            print(i, t_res[i, 1].item(), '\t', examples[i])
        print('----------------')


    wrong_eids = sorted(list(eid2wrongs))
    correct_eids = sorted(list(eid2corrects))
                

    if params.show_rel:
        for rel in sorted(rel2hits):
            hits = rel2hits[rel]
            total_triples_for_this_rel = rel2testcnt[rel]
            if params.head_tail_type:
                total_triples_for_this_rel //= 2
            print(int(rel in train_rels), rel, hits[1], '/', total_triples_for_this_rel, {f'hit@{k}':round(hits[k]/total_triples_for_this_rel, 3) for k in [1, 2]})

        for eid in correct_eids:
            # print(eid2relation[eid])
            print(examples[eid2pos[eid][0]])

    # target_eids =  sorted(target_eids)
    # correct_target_eids = [eid for eid in target_eids if eid in eid2corrects]
    ipdb.set_trace()
    if part:
        unseen_hits = Counter() # hit for each unseen
        unseen_hits_base = Counter() # base method hit for each unseen
        unseen_hits_cnt = 0 # total number unseen triples
        unseen_hits_rel_cnt = 0 # total number unseen relations
        unseen_hits_macro = Counter()
        unseen_hits_base_macro = Counter()

        for rel, hits in rel2hits.items():
            total_triples_for_this_rel = rel2testcnt[rel]
            if rel not in train_rels:
                unseen_hits_cnt += total_triples_for_this_rel
                unseen_hits_rel_cnt += 1
                for hit in [1, 2, 3, 4, 5, 10]:
                    unseen_hits[hit] += rel2hits[rel][hit]
                    unseen_hits_base[hit] += rel2hits_base[rel][hit]
                    unseen_hits_macro[hit] += rel2hits[rel][hit] / total_triples_for_this_rel
                    unseen_hits_base_macro[hit] += rel2hits_base[rel][hit] / total_triples_for_this_rel


        print(unseen_hits)
        print({f'hit@{k}':round(unseen_hits[k]/unseen_hits_cnt, 3) for k in [1, 2, 3, 4, 5, 10]})
        print({f'hit@{k}':round(unseen_hits_macro[k]/unseen_hits_rel_cnt, 3) for k in [1, 2, 3, 4, 5, 10]})
        print('----------------')
        print(unseen_hits_base)
        print({f'hit@{k}':round(unseen_hits_base[k]/unseen_hits_cnt, 3) for k in [1, 2, 3, 4, 5, 10]})
        print({f'hit@{k}':round(unseen_hits_base_macro[k]/unseen_hits_rel_cnt, 3) for k in [1, 2, 3, 4, 5, 10]})

    