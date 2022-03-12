TOTAL_UPDATES=30000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
LR=6e-04                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=5
MAX_SENTENCES=8        # Batch size.
NUM_NODES=1                                      # Number of machines
ROBERTA_PATH="mlm_nlp.pt"
CHECKPOINT_PATH="mlm_from_nlp_ke_of_imdb" #Directory to store the checkpoints
UPDATE_FREQ=`expr 736 / $NUM_NODES` # Increase the batch size
UPDATE_FREQ=64
DATA_DIR=data-bin/wikitext-103

#Path to the preprocessed KE dataset, each item corresponds to a data directory for one epoch
KE_DIR="kedata"
KE_DATA=$KE_DIR/IMDb2_0:$KE_DIR/IMDb2_1:$KE_DIR/IMDb2_2:$KE_DIR/IMDb2_3
KE_DATA="IMDb1"
DIST_SIZE=`expr $NUM_NODES \* 1`
#--bpe gpt2 --relation-desc --init-token 0
fairseq-train $DATA_DIR --arch roberta_base --num-workers 16 --bpe gpt2 --fix-batches-to-gpus --sample-break-mode complete --separator-token 2 \
        --KEdata $KE_DATA \
        --restore-file $ROBERTA_PATH \
        --save-dir $CHECKPOINT_PATH \
        --max-sentences $MAX_SENTENCES\
        --task MLMetKE --skip-invalid-size-inputs-valid-test --tokens-per-sample 512 --max-tokens 512\
        --required-batch-size-multiple 1 \
        --criterion OnlyKE \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --update-freq $UPDATE_FREQ \
        --negative-sample-size 1 \
        --ke-model TransE \
        --gamma 8 \
        --nrelation 28418  \
        --threshold-loss-scale 1  \
        --ddp-backend no_c10d --log-format tqdm --reset-optimizer --distributed-world-size ${DIST_SIZE} # --max-tokens 512 --distributed-port 23460 --distributed-world-size ${DIST_SIZE}
#       --arch roberta_base --sample-break-mode complete --separator-token 2 --reset-optimizer --distributed-port 23460 --distributed-world-size ${DIST_SIZE} --ddp-backend no_c10d \
#       --log-format simple --log-interval 1 ;
        #--relation-desc  #Add this option to encode the relation descriptions as relation embeddings (KEPLER-Rel in the paper)
#--skip-invalid-size-inputs-valid-test
#--tokens-per-sample 512 \
#--max-tokens 512 \

