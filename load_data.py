"""
Prepare data for BERTRL
"""
import os
import argparse
import logging
import re
from sys import path
import ipdb
from numpy.lib.npyio import load

from warnings import simplefilter

from collections import defaultdict, Counter
import networkx as nx
from torch import rand
from tqdm import tqdm, trange
import json
import random
import numpy as np

def flatten(list):
    return [_ for sublist in list for _ in sublist]


def show_path(path, entity2text):
    return [entity2text[e] for e in path]


def get_valid_paths(exclude_edge, paths):
    e1, e2, r_ind = exclude_edge
    # edge paths
    valid_paths = []
    for path in paths:
        if not any((e1, e2, r_ind) == edge for edge in path):
            valid_paths.append(path)
    return valid_paths


def is_in_G(e1, r, e2, G):
    if e1 in G and e2 in G and e2 in G[e1] and r in [r_dict['relation'] for r_dict in G[e1][e2].values()]:
        return True
    return False


def construct_local_entity2text(subgraph_entities, entity2text):
    local_text2entities = defaultdict(list)
    for e in subgraph_entities:
        local_text2entities[entity2text[e]].append(e)
    if len(local_text2entities) == len(subgraph_entities):
        local_entity2text = entity2text
    else:
        local_entity2text = {}
        for e in subgraph_entities:
            e_text = entity2text[e]
            if len(local_text2entities[e_text]) == 1:
                local_entity2text[e] = e_text
            else:
                local_entity2text[e] = e_text + ' ' + \
                    str(local_text2entities[e_text].index(e))
    return local_entity2text


def construct_subgraph_text(G, subgraph_entities, entity2text, relation2text, excluded=None):
    G_text = []
    # deal with repeat entities
    for e1, e2, r_dict in G.subgraph(subgraph_entities).edges.data():
        if (e1, r_dict['relation'], e2) == excluded:
            continue
        e1_text, e2_text = entity2text[e1], entity2text[e2]
        r_text = relation2text[r_dict['relation']]
        edge_text = f'{e1_text} {r_text} {e2_text};'
        G_text.append(edge_text)

    return G_text


def construct_paths_text(biG, valid_paths, entity2text, relation2text, edge_pattern='{} {} {};', conclusion_relation=None, params=None):
    downsample, use_reversed_order, sort_by_len = params.downsample, params.use_reversed_order, params.sort_by_len
    # edge_pattern = '“ {} ” {} “ {} ”;'
    # edge_pattern = '{} {} {};'
    # construct text from valid edge paths
    if sort_by_len:
        valid_paths = sorted(valid_paths, key=lambda x: len(x))

    G_text = []
    relation_paths = []
    rel2eids = defaultdict(list)
    for i, path in enumerate(valid_paths):
        relation_path = [conclusion_relation]
        for j, (e1, e2, r_ind) in enumerate(path):
            r_dict = biG[e1][e2][r_ind]
            e1_text, e2_text = entity2text[e1], entity2text[e2]
            r = r_dict['relation']
            if r.startswith('inv-'):
                if not use_reversed_order:
                    r_text = relation2text[r[4:]]
                    edge_text = edge_pattern.format(e2_text, r_text, e1_text)
                else:  # reversed order
                    r_text = 'inv- ' + relation2text[r[4:]]
                    edge_text = edge_pattern.format(e1_text, r_text, e2_text)
            else:
                r_text = relation2text[r]
                edge_text = edge_pattern.format(e1_text, r_text, e2_text)
            relation_path.append(r)

            G_text.append(edge_text)
        relation_path = tuple(relation_path)
        relation_paths.append(relation_path)
        rel2eids[relation_path].append(i)
        G_text.append('[SEP]')  # including a [SEP] at the end

    G_text = G_text[:-1]  # exluce last [SEP]
    if not G_text:
        return G_text

    G_text = ' '.join(G_text).split(' [SEP] ')

    if downsample:  # downsample the repeated relation paths
        sampled_eids = []
        for _, eids in rel2eids.items():
            sample_num_this_r = min(len(eids), 1)
            sampled_eids.extend(random.sample(eids, sample_num_this_r))
        G_text = [G_text[eid] for eid in sampled_eids]

    return G_text


def generate_bert_input_from_scratch(biG, set_type, triples, params=None):
    entity2text, relation2text = params.entity2text, params.relation2text

    question_pattern = 'Question: {} {} what ? Is the correct answer {} ?'

    valid_paths_cnter = Counter()
    fout = open(f'{params.bertrl_data_dir}/{set_type}.tsv', 'w')
    fout.write('# Quality\t#1 ID\t#2 ID\t#1 String\t#2 String\n')

    num_pos_examples = len(triples[set_type]['pos'])
    num_neg_samples_per_link = len(triples[set_type]['neg']) // num_pos_examples
    seen_neg = set()
    for i in trange(num_pos_examples):
        e1_pos, r_pos, e2_pos = triples[set_type]['pos'][i]
        paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_pos, target=e2_pos, cutoff=params.hop)]

        if set_type == 'train':
            assert len(paths) > 0  # at least have (e1, r, e2) in train graph

        if set_type == 'train':
            r_pos_inds = [k for k, r_dict in
                          biG[e1_pos][e2_pos].items() if r_dict['relation'] == r_pos]  # r_ind in biG local graph
            assert len(r_pos_inds) == 1
            r_pos_ind = r_pos_inds[0]
            valid_paths = get_valid_paths((e1_pos, e2_pos, r_pos_ind), paths)
        else:
            r_pos_ind = None
            valid_paths = paths

        if valid_paths:
            valid_paths_cnter['valid_paths'] += 1

        # [('/m/0hvvf', '/m/039bp', 0)]
        # [('/m/0hvvf', '/m/07s9rl0', 0), ('/m/07s9rl0', '/m/0m9p3', 0), ('/m/0m9p3', '/m/039bp', 0)]
        subgraph_entities_pos = flatten([edge[:2] for edge in flatten(valid_paths)])
        subgraph_entities_pos = list(set(subgraph_entities_pos))

        local_entity2text = construct_local_entity2text(subgraph_entities_pos, entity2text)
        e1_text_pos, e2_text_pos = local_entity2text[e1_pos], local_entity2text[e2_pos]
        r_text_pos = relation2text[r_pos]

        conclusion_pos = question_pattern.format(e1_text_pos, r_text_pos, e2_text_pos)

        if params.subgraph_input:
            G_text_pos_edges = construct_subgraph_text(
                biG, subgraph_entities_pos, local_entity2text, relation2text, excluded=(e1_pos, r_pos, e2_pos))
            G_text_pos_edges = ['  '.join(G_text_pos_edges)]
        else:
            # now use path text
            G_text_pos_edges = construct_paths_text(biG, valid_paths, local_entity2text, relation2text, conclusion_relation=r_pos, params=params)

        def shuffle_G_text_edges(G_text_edges):
            shuffled_G_text = []
            for G_text in G_text_edges:
                G_edges = G_text.strip(';').split('; ')
                random.shuffle(G_edges)
                shuffled_G_text.append('; '.join(G_edges) + ';')
            return shuffled_G_text

        if params.shuffle_times > 0:
            G_text_pos_edges = shuffle_G_text_edges(G_text_pos_edges)

        if G_text_pos_edges and params.combine_paths:
            G_text_pos_edges = [' [SEP] '.join(G_text_pos_edges[:params.kept_paths])]

        # sample/take positive path for train/test
        for ii, G_text_pos in enumerate(G_text_pos_edges):
            context_pos = 'Context: {}'.format(G_text_pos)
            if params.block_body:
                context_pos = ''

            if ii >= params.kept_paths and set_type == 'train':  # drop some paths in training
                break

            if G_text_pos:
                fout.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    1, set_type+'-pos-'+str(i), 'train-pos-'+str(i), conclusion_pos, context_pos))
                valid_paths_cnter[1] += 1

        if set_type == 'train':
            # sampling negative pairs for train
            # this always negative sampling from the neighbors
            pairs = []
            e1_neigh_to_dis = nx.single_source_shortest_path_length(biG, e1_pos, cutoff=3)  # 3 
            e2_neigh_to_dis = nx.single_source_shortest_path_length(biG, e2_pos, cutoff=3)

            common_neighs = set(e1_neigh_to_dis) & set(e2_neigh_to_dis)

            e1_neigh_to_dis = {k: v for k, v in e1_neigh_to_dis.items() if k in common_neighs}
            e2_neigh_to_dis = {k: v for k, v in e2_neigh_to_dis.items() if k in common_neighs}

            for neigh in e1_neigh_to_dis:
                if (e1_pos, r_pos, neigh) not in seen_neg and neigh not in (e1_pos, e2_pos):
                    # exclude the sampled negative in training positive
                    # if neigh in G[e1_pos] and r_pos in [r_dict['relation'] for r_dict in G[e1_pos][neigh].values()]:  # (e1_pos, r_pos, neigh) in G train
                    if is_in_G(e1_pos, r_pos, neigh, biG): # (e1_pos, r_pos, neigh) in G train
                        valid_paths_cnter['in_train'] += 1
                        continue

                    pairs.append((e1_pos, neigh))
                    seen_neg.add((e1_pos, r_pos, neigh))
            for neigh in e2_neigh_to_dis:
                if (neigh, r_pos, e2_pos) not in seen_neg and neigh not in (e1_pos, e2_pos):
                    
                    # if e2_pos in G[neigh] and r_pos in [r_dict['relation'] for r_dict in G[neigh][e2_pos].values()]:  # (neigh, r_pos, e2_pos) in G train
                    if is_in_G(neigh, r_pos, e2_pos, biG): # (neigh, r_pos, e2_pos) in G train
                        valid_paths_cnter['in_train'] += 1
                        continue
                    pairs.append((neigh, e2_pos))
                    seen_neg.add((neigh, r_pos, e2_pos))

            # make sure there is a path 3 length path
            # if not pairs:
            #     continue

            if G_text_pos_edges:
                pairs = random.sample(pairs, min(len(pairs), params.neg)) # neg = 10
            else:
                pairs = random.sample(pairs, min(len(pairs), 1))

            for j, (e1_neg, e2_neg) in enumerate(pairs):
                paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_neg, target=e2_neg, cutoff=params.hop)]

                valid_paths = get_valid_paths((e1_pos, e2_pos, r_pos_ind), paths)  # in train remove all path contains e1_pos, r_pos, e2_pos,
                # why also do this for negative examples?
                assert biG[e1_pos][e2_pos][r_pos_ind]['relation'] == r_pos

                subgraph_entities_neg = flatten([edge[:2] for edge in flatten(valid_paths)])
                subgraph_entities_neg = list(set(subgraph_entities_neg))

                local_entity2text = construct_local_entity2text(subgraph_entities_neg, entity2text)
                e1_text_neg, e2_text_neg = local_entity2text[e1_neg], local_entity2text[e2_neg]
                r_text_neg = r_text_pos

                if params.subgraph_input:
                    G_text_neg_edges = construct_subgraph_text(
                        G, subgraph_entities_neg, local_entity2text, relation2text, excluded=(e1_pos, r_pos, e2_pos))
                    G_text_neg_edges = ['  '.join(G_text_neg_edges)]
                else:
                    G_text_neg_edges = construct_paths_text(biG, valid_paths, local_entity2text, relation2text, conclusion_relation=r_pos, params=params)


                if params.shuffle_times > 0:
                    G_text_neg_edges = shuffle_G_text_edges(G_text_neg_edges)

                if G_text_neg_edges and params.combine_paths:
                    G_text_neg_edges = [' [SEP] '.join(G_text_neg_edges[:params.kept_paths])]

                for jj, G_text_neg in enumerate(G_text_neg_edges):
                    if jj >= params.kept_paths and set_type == 'train':  # drop some paths in training for neg
                        break
                    conclusion_neg = question_pattern.format(e1_text_neg, r_text_neg, e2_text_neg)
                    context_neg = 'Context: {}'.format(G_text_neg)
                    if params.block_body:
                        context_neg = ''
                    # context_neg, conclusion_neg = conclusion_neg, context_neg
                    fout.write('{}\t{}\t{}\t{}\t{}\n'.format(0, set_type+'-neg-'+str(i)+'-'+str(
                        j), set_type+'-neg-'+str(i)+'-'+str(j), conclusion_neg, context_neg))
                    valid_paths_cnter[0] += 1

        elif set_type == 'test' or set_type == 'dev':
            # take pre-generated ranking head/tail triplets
            num_empty_path_neg_this_pos = 0
            for j in range(num_neg_samples_per_link):  # pos i 's jth neg
                """
                    pbar = tqdm(total=len(pos_edges))
                    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
                        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(
                            pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
                    pos1' neg, pos2's neg, ...
                """
                e1_neg, r_neg, e2_neg = triples[set_type]['neg'][i * num_neg_samples_per_link + j]

                paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_neg, target=e2_neg, cutoff=params.hop)]  # be careful to choose biG

                valid_paths = paths
                if len(valid_paths) == 0:
                    num_empty_path_neg_this_pos += 1
                    continue

                subgraph_entities_neg = flatten([edge[:2] for edge in flatten(paths)])
                subgraph_entities_neg = list(set(subgraph_entities_neg))

                local_entity2text = construct_local_entity2text(subgraph_entities_neg, entity2text)
                e1_text_neg, e2_text_neg = local_entity2text[e1_neg], local_entity2text[e2_neg]
                r_text_neg = relation2text[r_neg]

                conclusion_neg = question_pattern.format(e1_text_neg, r_text_neg, e2_text_neg)

                # G_text_neg_edges = construct_subgraph_text(
                # G, subgraph_entities_neg, local_entity2text, relation2text, excluded=(e1_pos, r_pos, e2_pos), join_edge_text=False)

                if params.subgraph_input:
                    G_text_neg_edges = construct_subgraph_text(
                        biG, subgraph_entities_neg, local_entity2text, relation2text, excluded=(e1_pos, r_pos, e2_pos))
                    G_text_neg_edges = ['  '.join(G_text_neg_edges)]
                else:
                    # now use path text
                    G_text_neg_edges = construct_paths_text(biG, valid_paths, local_entity2text, relation2text, conclusion_relation=r_neg, params=params)

                if params.shuffle_times > 0:
                    G_text_neg_edges = shuffle_G_text_edges(G_text_neg_edges)

                if G_text_neg_edges and params.combine_paths:
                    G_text_neg_edges = [' [SEP] '.join(G_text_neg_edges[:params.kept_paths])]

                # G_text_neg = ' '.join(G_text_neg_edges)
                for G_text_neg in G_text_neg_edges:
                    context_neg = 'Context: {}'.format(G_text_neg)
                    if params.block_body:
                        context_neg = ''
                    if G_text_neg:
                        fout.write('{}\t{}\t{}\t{}\t{}\n'.format(
                            0, set_type+'-neg-'+str(i)+'-'+str(j), set_type+'-neg-'+str(i)+'-'+str(j), conclusion_neg, context_neg))
                        valid_paths_cnter[0] += 1

    print('# statistics: ', valid_paths_cnter)
    fout.close()

def load_train(params):
    # construct graph
    triples = {'train': defaultdict(list), 'valid': defaultdict(
        list), 'test': defaultdict(list)}


    biG = nx.MultiDiGraph()
    # all postive train
    with open(f'{params.main_dir}/data/{params.dataset}/train_{params.part}.txt') as fin:
        for l in fin:
            e1, r, e2 = l.strip().split('\t')
            triples[params.set_type]['pos'].append([e1, r, e2])
            # bidirectional G
            biG.add_edges_from([(e1, e2, dict(relation=r))])
            biG.add_edges_from([(e2, e1, dict(relation='inv-'+r))])


    generate_bert_input_from_scratch(biG, 'train', triples, params=params)


def load_test(params):

    # construct graph
    triples = {'train': defaultdict(list), 'test': defaultdict(list)}

    biG = nx.MultiDiGraph()
    # all train
    with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/train.txt') as fin:  # fb237_ind, use full inductive test fact graph
        for l in fin:
            e1, r, e2 = l.strip().split('\t')
            triples['train']['pos'].append([e1, r, e2])
            # bidirectional G
            biG.add_edges_from([(e1, e2, dict(relation=r))])
            biG.add_edges_from([(e2, e1, dict(relation='inv-'+r))])

    # load from unified bertrl generated test
    num_samples = params.candidates
    for head_type in ['head', 'tail']:
        with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/ranking_{head_type}.txt') as fin:  # fb237_ind
            for i, l in enumerate(fin):
                e1, r, e2 = l.strip().split('\t')
                if i % num_samples == 0:
                    triples[params.set_type]['pos'].append([e1, r, e2])
                else:
                    triples[params.set_type]['neg'].append([e1, r, e2])

    generate_bert_input_from_scratch(biG, 'test', triples, params=params)


def load_valid(params):

    # construct graph
    triples = {'train': defaultdict(list), 'dev': defaultdict(list)}

    biG = nx.MultiDiGraph()
    # all train
    with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/train.txt') as fin:  # fb237v1_ind, use full inductive test fact graph
        for l in fin:
            e1, r, e2 = l.strip().split('\t')
            triples['train']['pos'].append([e1, r, e2])
            # bidirectional G
            biG.add_edges_from([(e1, e2, dict(relation=r))])
            biG.add_edges_from([(e2, e1, dict(relation='inv-'+r))])

    all_entities = list(biG)
    for head_type in ['head', 'tail']:
        with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/valid.txt') as fin:  # fb237v1_ind, use full inductive test fact graph
            for l in fin:
                e1_pos, r_pos, e2_pos = l.strip().split('\t')
                triples['dev']['pos'].append([e1_pos, r_pos, e2_pos])  # totally append two times pos

                # sampling negative pairs
                pairs_unreached = []
                if head_type == 'head':
                    j = 0
                    while j < params.candidates:
                        e2_neg = random.choice(all_entities)
                        if not is_in_G(e1_pos, r_pos, e2_neg, biG):
                            triples['dev']['neg'].append([e1_pos, r_pos, e2_neg])
                            j += 1
                else:
                    j = 0
                    while j < params.candidates:
                        e1_neg = random.choice(all_entities)
                        if not is_in_G(e1_neg, r_pos, e2_pos, biG):
                            triples['dev']['neg'].append([e1_neg, r_pos, e2_pos])
                            j += 1

    generate_bert_input_from_scratch(biG, 'dev', triples, params=params)


def main(params):

    params.main_dir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
    params.dataset_short = params.dataset.split('_')[0]  # without suffix
    params.bertrl_data_dir = f'{params.main_dir}/bertrl_data/{params.dataset}_hop{params.hop}_{params.part}{params.suffix}'

    if params.dataset.startswith('WN18RR'):
        params.text_data_dir = f'{params.main_dir}/data/text/WN18RR/'
    elif params.dataset.startswith('fb'):
        params.text_data_dir = f'{params.main_dir}/data/text/FB237/'
    elif params.dataset.startswith('nell'):
        params.text_data_dir = f'{params.main_dir}/data/text/NELL995/'
    else:
        assert 0

    entity2text = {}
    # entity2longtext = {}
    with open(f'{params.text_data_dir}/entity2text.txt') as fin:
        for l in fin:
            entity, text = l.strip().split('\t')
            name = text.split(',')[0]
            entity2text[entity] = name
            # entity2longtext[entity] = text
    relation2text = {}
    with open(f'{params.text_data_dir}/relation2text.txt') as fin:
        for l in fin:
            relation, text = l.strip().split('\t')
            relation2text[relation] = text

    params.entity2text, params.relation2text = entity2text, relation2text
    # params.entity2longtext = entity2longtext

    if params.block_body:
        params.bertrl_data_dir += '_block_body'

    if not os.path.exists(params.bertrl_data_dir):
        os.makedirs(params.bertrl_data_dir)

    if params.set_type == 'train':
        load_train(params)
    elif params.set_type == 'test':
        load_test(params)
    elif params.set_type == 'dev':
        load_valid(params)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='BERTRL model')
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--part", type=str, default="full",
                        help="part")

    # Data processing pipeline params
    parser.add_argument("--hop", type=int, default=3,
                        help="max reasoning path length")
    parser.add_argument('--set_type', '-st', type=str, default='train',
                        help='set type of train/valid/test')
    parser.add_argument("--shuffle_times", type=int, default=0,
                        help="Shuffle times")
    parser.add_argument("--kept_paths", type=int, default=3,
                        help="number of kept sub paths")

    parser.add_argument("--suffix", type=str, default="",
                        help="additional suffix")
    parser.add_argument("--downsample", default=False, action='store_true',
                        help="downsample or not")
    parser.add_argument("--block_body", default=False, action='store_true',
                        help="block body or not")
    parser.add_argument("--ind_suffix", type=str, default='_ind',
                        help="ind suffix")
    parser.add_argument("--use_reversed_order", default=False, action='store_true',
                        help="use reversed order or not")
    parser.add_argument("--sort_by_len", default=False, action='store_true',
                        help="sort_by_len ")
    parser.add_argument("--combine_paths", default=False, action='store_true',
                        help="combine_paths ")
    parser.add_argument("--subgraph_input", default=False, action='store_true',
                        help="subgraph_input ")

    parser.add_argument("--neg", type=int, default=10,
                        help="neg")
    parser.add_argument("--candidates", type=int, default=50,
                        help="number of candidates for ranking")

    params = parser.parse_args()
    main(params)
