CONFIG=$1
GPUS=$2
PORT=${PORT:-31000}

echo $PYTHONPATH
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4} --seed 1973595181