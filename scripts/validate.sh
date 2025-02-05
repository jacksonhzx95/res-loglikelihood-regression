set -x

CONFIG=$1
CKPT=$2
PORT=${4:2145}

HOST=$(hostname -i)

python ./scripts/validate_scoliosis.py \
    --cfg ${CONFIG} \
    --valid-batch 16 \
    --checkpoint ${CKPT} \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
