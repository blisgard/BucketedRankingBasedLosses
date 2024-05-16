CONFIG=$1
GPUS=$2
PORT=${PORT:-38500}

echo $PYTHONPATH
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4} --resume-from /truba/home/feyavuz/ranksortloss/Co-DETR/work_dirs/bucketed_co_dino_5scale_r50_1x_coco/epoch_9.pth