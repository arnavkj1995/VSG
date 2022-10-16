#!/bin/bash

wandb offline
wand_val=False

MODEL=$1
SUFFIX=$2
ENV_TYPE=$3
TL=$4
ENV_NAME=bringbackshapes_${ENV_TYPE}
LOG_DIR=logs/bbs/
STEPS=2503001

# Params
MAX_DISTRACTORS=$5
MAX_OBJECTS=$6
VARIABLE_NUM_OBJECTS=$7
VARIABLE_NUM_DISTRACTORS=$8
VARIABLE_GOAL_POSITION=$9
AGENT_VIEW_SIZE=${10}
ARENA_SCALE=${11}
GATE_PRIOR=${12}
SEED=${13}

echo $ENV_NAME
echo $MODEL

EXTRA_ARGS=" --time_limit ${TL} --max_distractors ${MAX_DISTRACTORS} --max_objects ${MAX_OBJECTS} --variable_num_objects ${VARIABLE_NUM_OBJECTS} --variable_num_distractors ${VARIABLE_NUM_DISTRACTORS} --variable_goal_position ${VARIABLE_GOAL_POSITION} --agent_view_size ${AGENT_VIEW_SIZE} --arena_scale ${ARENA_SCALE}"

if [ "$MODEL" == "DreamerV1" ];
then
    ############################################## DreamerV1 #############################################
    CONFIG='bringbackshapes bringbackshapes_gaussian'
    ID=DreamerV1_${SUFFIX}

elif [ "$MODEL" == "DreamerV2" ];
then
    ############################################# DreamerV2 ##############################################
    CONFIG='bringbackshapes'
    ID=DreamerV2_${SUFFIX}

elif [ "$MODEL" == "VSG" ];
then
    ############################################# VSG (C) ##############################################
    CONFIG='bringbackshapes'
    ID=VSGC_prior${GATE_PRIOR}_${SUFFIX}
    EXTRA_ARGS="--dyn_gate_prior ${GATE_PRIOR} --dyn_gate_scale 0.1 --dyn_cell sgru ${EXTRA_ARGS}"

elif [ "$MODEL" == "SVSG" ];
then
    ############################################# SVSG ################################################
    DIM=1024
    CONFIG='bringbackshapes bringbackshapes_svsg'
    ID=SVSG_dim${DIM}_prior${GATE_PRIOR}_${SUFFIX}
    EXTRA_ARGS="--dyn_gate_prior ${GATE_PRIOR} --dyn_gate_scale 0.1 --dyn_cell sgru ${EXTRA_ARGS}"

else
    echo $"Invalid Model"
    exit 1
fi

python dreamer.py --logdir ${LOG_DIR} --configs defaults $CONFIG --use_wandb ${wand_val} --task ${ENV_NAME} --steps $STEPS --seed $SEED --id $ID ${EXTRA_ARGS}
