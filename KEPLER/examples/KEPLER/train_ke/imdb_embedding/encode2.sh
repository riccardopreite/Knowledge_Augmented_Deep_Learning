mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
python -m examples.roberta.multiprocessing_bpe_encoder \
		--encoder-json gpt2_bpe/encoder.json \
		--vocab-bpe gpt2_bpe/vocab.bpe \
		--inputs Qdesc2.txt \
		--outputs Qdesc2.bpe \
		--keep-empty \
		--workers 60
