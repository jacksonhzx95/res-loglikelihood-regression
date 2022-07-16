set -x

CONFIG=$1
CKPT=$2
PORT=${3:-23455}

HOST=$(hostname -i)

python ./scripts/validate_scoliosis_offline.py \
    --cfg ${CONFIG} \
    --valid-batch 1 \
    --checkpoint ${CKPT} \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
