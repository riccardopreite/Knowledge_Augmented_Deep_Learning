wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref Qdesc2train.bpe \
    --validpref Qdesc2valid.bpe \
    --destdir data-bin/imdb \
    --workers 60
