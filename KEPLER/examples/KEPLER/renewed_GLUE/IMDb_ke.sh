TOTAL_NUM_UPDATES=7812
WARMUP_UPDATES=469
LR=1e-05
NUM_CLASSES=3
MAX_SENTENCES=8        # Batch size.
SLASH="/"
ROBERTA_PATH="${1:-ke.pt}"
NAME="${ROBERTA_PATH/.pt/''}"
NAME="${NAME/$SLASH/'-'}"
#--load-checkpoint-heads \
fairseq-train IMDb_ke-bin/ --load-checkpoint-heads\
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --save-dir IMDb_ke-$NAME-ckpt \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --threshold-loss-scale 1 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --truncate-sequence \
    --find-unused-parameters --ddp-backend=no_c10d\
    --update-freq 4
