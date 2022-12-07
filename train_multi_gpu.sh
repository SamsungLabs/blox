#!/bin/bash

next_model=${1:-0}
models_per_gpu=${2:-25}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


GPUS=/horovod/generated/hostfile
if [ ! -f "$GPUS" ]; then
    echo "Horovod hostfile does not exist, falling back to using localhost"
    GPUS=(0 1 2 3 4 5 6 7)
    IS_DIST=0
else
    CHILDREN_ADDRS=($(cat $GPUS | awk '{print $1}' | tr '\n' ' '))
    GPUS=(${!CHILDREN_ADDRS[@]})
    echo "Horovod file found!"
    echo "${CHILDREN_ADDRS[@]}"
    IS_DIST=1
fi


declare -a PIDS

function clean_up {
    for pid in ${pids[*]}; do
        kill $pid
    done
    exit
}
trap clean_up SIGHUP SIGINT SIGTERM


for ((i=0; i<${#GPUS[@]}; i++)) do
    ((last_model=next_model+models_per_gpu))
    if [ $IS_DIST -eq 1 ]; then
        ADDR=${CHILDREN_ADDRS[i]}
        CMD=(ssh "$ADDR" "cd $SCRIPT_DIR; ./train_bucket.sh $next_model $last_model $selective --gpus 0 ${@:3}")
    else
        GPU=${GPUS[i]}
        CMD=("./train_bucket.sh" "$next_model" "$last_model" "--gpus" "$GPU" "${@:3}")
    fi
    echo "${CMD[@]}"
    "${CMD[@]}" &
    PIDS[i]=$!
    ((next_model=last_model))
done

echo "Next model to launch: $next_model"
for pid in ${PIDS[*]}; do
    wait $pid
done
