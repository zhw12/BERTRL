# Stratified Sampling, keep all relations
import numpy as np
import ipdb
from collections import defaultdict
import random
import math


dataset = 'fb237'
example_ids = []
relations = []
rel2eids = defaultdict(list)

data = []
with open(f'./data/{dataset}/train.txt') as fin:
    for i, l in enumerate(fin):
        e1, r, e2 = l.strip().split('\t')
        example_ids.append(i)
        relations.append(r)
        data.append(l)
        rel2eids[r].append(i)

num_samples = 5000
sample_ratio = float(num_samples) / len(example_ids)

sampled_eids = []

for r, eids in rel2eids.items():
    sample_num_this_r = max(int(round(sample_ratio * len(eids))), 1)
    sampled_eids.extend(random.sample(eids, sample_num_this_r))

sampled_eids = sorted(sampled_eids)


with open(f'./data/{dataset}/train_{num_samples}.txt', 'w') as fout:
    for eid in sampled_eids:
        fout.write(data[eid])
        
