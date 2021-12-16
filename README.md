# BERTRL: Inductive Relation Prediction by BERT
Code and data for AAAI2022 paper [Inductive Relation Prediction by BERT](https://arxiv.org/pdf/2103.07102.pdf), which aims to study the problem of exploiting structural and textual information in knowledge graph completion leverging pre-trained langauge models. BERTRL feeds texts of candidate triple instances and their possible reasoning paths to BERT and predicts the existence of the triple.

## Requirements:
- [huggingface transformer 3.3.1](https://github.com/huggingface/transformers)
- [pytorch 1.5.0](https://pytorch.org/)
- networkx 2.5
- tqdm

## Download the Dataset Split
Here we provide the data split used in paper in folder "data". The $DATASET$PART and $DATASET$PART_ind contain corresponding transductive and inductive subgraphs. 
Each train/valid/test file contains a list of knowledge graph triples. "ranking_head.txt" and "ranking_tail.txt" are presampled candidates 
for the predicting the missing tail triple and missing head triple in knowledge graph completion. Each triple contains 50 candidates for tail and 50 for head in this file.
$DATASET denotes the dataset name, and $PART denotes the size of the dataset, whether it is a fewshot version or full. If $PART is not specified, it is full by default.


## Preprocessing Data
folder "bertrl_data" provides an example of preprocessed data to be input of BERTRL model. 
They are actual tsv data examples required for BERTRL. Here we show the example preprocessing scripts. $DATASET denotes the name of the dataset in folder "data", e.g. fb237.
part paramerter can be specified as full, 1000, 2000, referring to folder "data".

```
python load_data.py -d $DATASET -st train --part full --hop 3 --ind_suffix "_ind" --suffix "_neg10_path3_max_inductive"
python load_data.py -d $DATASET -st test --part full --hop 3 --ind_suffix "_ind" --suffix "_neg10_path3_max_inductive"
```

## BERTRL
1. Training model
We provide example bash scripts in train.bash

2. Evaluating model
We provide example bash scripts in test.bash.
This generates the BERTRL scoring for each of its examples. Then we evaluate the final ranking results by additional script aggregating the results. Here is an example.
python eval_bert.py -d ${DATASET}_hop3_full_neg10_path3_max_inductive

3. Hyperpameters
Our provided training and evaluating scirpts use the hyperparameters mentioned in the paper. Check our paper for more details.

## Baselines
We use code provided by authors for baselines in our experiments.
RuleN is a ruled-based method. Detailed instructions can be found [here](http://web.informatik.uni-mannheim.de/RuleN/).
GraIL is subgraph reasoning method. Detailed instructions can be found [here](https://github.com/kkteru/grail).

## Citation
If you find this project useful, please cite it using the following format


    @article{zha2021inductive,
    title={Inductive Relation Prediction by BERT},
    author={Zha, Hanwen and Chen, Zhiyu and Yan, Xifeng},
    journal={arXiv preprint arXiv:2103.07102},
    year={2021}
    }
## Q&A
If you have any questions about the paper and the github, please feel free to leave an issue or send me an email.