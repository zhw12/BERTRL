""" Finetuning BERTRL for scoring.
Adapted from huggingface transformers sequence classification on GLUE"""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

import torch
from sklearn.metrics import average_precision_score
from collections import defaultdict, Counter
import itertools

from model import BertForKBCSequenceClassification



logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )

    # model = AutoModelForSequenceClassification.from_pretrained(
    model = BertForKBCSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )

    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    # evaluating BERTRL predictions in Hits and MRR.  
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_ranking_metrics_fn(p: EvalPrediction):
            t_res = torch.softmax(torch.from_numpy(p.predictions), 1)
            eid2pos = {}
            eid2neg = defaultdict(list)

            bertrl_data_file = f'{data_args.data_dir}/dev.tsv' # bertrl data file (linearized)
            examples = [l.strip().split('\t') for l in open(bertrl_data_file)][1:]

            # another set of evaluation
            eid2pos = defaultdict(list)
            eid2neg = defaultdict(list)
            eid2neg_lv2 = {}
            for i, example in enumerate(examples):
                label = int(example[0])
                eid = example[1].split('-')[2] # test-neg-2-24
                if label == 1:
                    eid2pos[eid].append(i)
                else:
                    e_negid = example[1].split('-')[3] # test-neg-2-24
                    eid2neg[eid].append(i)
                    if eid not in eid2neg_lv2:
                        eid2neg_lv2[eid] = defaultdict(list)
                    eid2neg_lv2[eid][e_negid].append(i)

            
            hit1 = 0 
            # hits = {1:0, 2:0, 3:0, 4:0, 5:0, 10:0, 20:0, 30:0, 40:0, 50:0}
            hits = Counter()
            ranks = []
            for eid, pos_is in eid2pos.items():
                pos_scores = t_res[pos_is, 1]

                neg_is = eid2neg[eid]
                geq_j = 1  # rank
                if not neg_is:  # no negative, only positive
                    hit1 += 1
                else:
                    neg_scores_of_eid = []
                    neg_scores_lists = []
                    for neg_eid in eid2neg_lv2[eid]:
                        neg_is_of_eid = eid2neg_lv2[eid][neg_eid]
                        neg_scores_ = t_res[neg_is_of_eid, 1]  # previously a bug here, previously use neg_is which lowers the hit 2+ performance
                        neg_max_score_ = torch.max(neg_scores_).item()
                        neg_scores_of_eid.append(neg_max_score_)
                        neg_scores_lists.append(neg_scores_.sort(0, descending=True).values.tolist())

                    _scores_pos = pos_scores.sort(0, descending=True).values.tolist()
                    for _scores in neg_scores_lists:
                        for s1, s2 in itertools.zip_longest(_scores_pos, _scores, fillvalue=100): # fill 100 as a default value
                            if s1 < s2 or s1 == 100:
                                geq_j += 1
                                break
                            elif s1 == s2:
                                continue
                            else:
                                break
                    
                for hit in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]:
                    if geq_j <= hit:
                        hits[hit] += 1
                ranks.append(geq_j)

            hits = {f'hit@{k}':hits[k] for k in sorted(hits)}
            mrr = np.mean(1 / np.array(ranks))

            hits['filtered_mrr'] = mrr

            return hits

        def compute_metrics_fn(p: EvalPrediction):
            # print(compute_ranking_metrics_fn(p))
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            glue_metrics = glue_compute_metrics(task_name, preds, p.label_ids)
            glue_metrics.update(compute_ranking_metrics_fn(p))
            return glue_metrics
        # return compute_ranking_metrics_fn
        return compute_metrics_fn


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # customized early stopping
    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            # ranking evaluations
            # preds = trainer.predict(test_dataset=eval_dataset).predictions
 
            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                # write scores of each example into files
                with open(f'{training_args.output_dir}/test_results_prediction_scores.npy', 'wb') as fout:
                    np.save(fout, predictions)
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results

if __name__ == "__main__":
    main()