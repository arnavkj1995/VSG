#!/bin/bash

wandb offline

ENV_NAME=$1
MODEL=$2
SUFFIX=$3
GATE_PRIOR=$4
DIM=$5
SEED=$6

STEPS=500005
CONFIG=dmc
use_wandb=False
LOG_DIR='logs/dmc/'

EXTRA_ARGS=""

if [ "$MODEL" == "DreamerV1" ];
then
    ############################################## DreamerV1 #############################################
    EXTRA_ARGS="${EXTRA_ARGS} --dyn_deter ${DIM} --dyn_hidden ${DIM}"
    ID=DreamerV1_${SUFFIX}

elif [ "$MODEL" == "DreamerV2" ];
then
    ############################################# DreamerV2 ##############################################
    ID=DreamerV2_${SUFFIX}
    EXTRA_ARGS="${EXTRA_ARGS} --dyn_deter ${DIM} --dyn_hidden ${DIM} --dyn_discrete 32"

elif [ "$MODEL" == "VSG" ];
then
    ############################################# VSG (C) ##############################################
    ID=VSGC_prior${GATE_PRIOR}_${SUFFIX}
    EXTRA_ARGS="--dyn_deter ${DIM} --dyn_hidden ${DIM} --dyn_gate_prior ${GATE_PRIOR} --dyn_gate_scale 0.1 --dyn_cell sgru --dyn_discrete 32 ${EXTRA_ARGS}"

elif [ "$MODEL" == "SVSG" ];
then
    ############################################# SVSG ################################################
    ID=SVSG_prior${GATE_PRIOR}_${SUFFIX}
    EXTRA_ARGS="--dyn_deter ${DIM} --dyn_stoch ${DIM} --dyn_hidden ${DIM} --dyn_kl_mask True --save_eps False --model_type srssm --dyn_gate_prior ${GATE_PRIOR} --dyn_gate_scale 0.1 --dyn_cell sgru ${EXTRA_ARGS}"

else
    echo $"Invalid Model"
    exit 1
fi

python dreamer.py --logdir ${LOG_DIR} --configs defaults $CONFIG --use_wandb ${use_wandb} --task dmc_${ENV_NAME} --steps $STEPS --seed $SEED --id $ID ${EXTRA_ARGS}
