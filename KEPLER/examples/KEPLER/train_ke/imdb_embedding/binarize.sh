wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
for KE_Data in ./IMDb2/; do \
    for SPLIT in head tail negHead negTail; do \
        fairseq-preprocess \
            --only-source \
            --srcdict gpt2_bpe/dict.txt \
            --trainpref ${KE_Data}${SPLIT}/train.bpe \
            --validpref ${KE_Data}${SPLIT}/valid.bpe \
            --destdir ${KE_Data}${SPLIT} \
            --workers 60; \
    done \
done

##if fairseq-preprocess cannot be founded, use "python -m fairseq_cli.preprocess" instead
