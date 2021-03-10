# #!/usr/bin/env bash

DATA_DIR="../data"
TRAIN_DIR="$DATA_DIR/$1"

SUFFIX="$3"
TRAIN_FILE="train${SUFFIX}.txt"

LEARNED_RULES="learned_rules.txt${SUFFIX}"

#Learn rules with path length
java -cp RuleN.jar de.unima.ki.arch.LearnRules -t $TRAIN_DIR/$TRAIN_FILE -s1 1000 -s2 1000 -p $2 -o $TRAIN_DIR/$LEARNED_RULES
