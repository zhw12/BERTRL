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


rel2cnt = {}
for rel in rel2eids:
    rel2cnt[rel] = len(rel2eids[rel])

num_subset_relations = 50
seen_rels = set()
while len(seen_rels) < num_subset_relations:
    rel = random.choices(list(rel2cnt), list(rel2cnt.values()), k=1)[0]
    if rel not in seen_rels:
        seen_rels.add(rel)

sampled_eids = []
for rel in seen_rels:
    sampled_eids.extend(rel2eids[rel])


with open(f'./data/{dataset}/train_rel{num_subset_relations}.txt', 'w') as fout:
    for eid in sampled_eids:
        fout.write(data[eid])
