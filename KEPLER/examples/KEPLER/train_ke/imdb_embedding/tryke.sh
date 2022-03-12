TOTAL_UPDATES=50000    # Total number of training steps
WARMUP_UPDATES=6000    # Warmup the learning rate over this many updates
LR=6e-04                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=1        # Batch size.
NUM_NODES=1					 # Number of machines
ROBERTA_PATH="ke.pt" #Path to the original roberta model
CHECKPOINT_PATH="ke_attempt" #Directory to store the checkpoints
UPDATE_FREQ=`expr 3136 / $NUM_NODES` # Increase the batch size

DATA_DIR=data-bin/wikitext-103

#Path to the preprocessed KE dataset, each item corresponds to a data directory for one epoch
KE_DIR="kedata"
#KE_DATA=$KE_DIR/IMDb2_0:$KE_DIR/IMDb2_1:$KE_DIR/IMDb2_2:$KE_DIR/IMDb2_3:
KE_DATA=IMDb2
DIST_SIZE=`expr $NUM_NODES \* 2`

fairseq-train $DATA_DIR \
        --KEdata $KE_DATA \
        --restore-file $ROBERTA_PATH \
        --save-dir $CHECKPOINT_PATH \
        --max-sentences $MAX_SENTENCES \
        --tokens-per-sample 512 \
        --max-tokens 512 \
        --task MLMetKE \
        --sample-break-mode complete \
        --required-batch-size-multiple 1 \
        --arch roberta_base \
        --criterion MLMetKE \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --update-freq $UPDATE_FREQ \
        --negative-sample-size 1 \
        --ke-model TransE \
        --init-token 0 \
        --separator-token 2 \
        --gamma 4 \
        --nrelation 822 \
        --threshold-loss-scale 1 \
        --reset-optimizer --distributed-port 23462 --distributed-world-size ${DIST_SIZE} --ddp-backend no_c10d \
        --log-format simple --log-interval 1 ;
        #--relation-desc  #Add this option to encode the relation descriptions as relation embeddings (KEPLER-Rel in the paper)
