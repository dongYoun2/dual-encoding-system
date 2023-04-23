if [ "$1" = "vocab" ]; then
	python vocab.py data/msrvtt10k/TextData/msrvtt10ktrain.caption.txt vocab.json
elif [ "$1" = "word2vec" ]; then
	PYTHONPATH=. python scripts/word2vec.py pretrained_weight.npy --vocab_file=vocab.json --word2vec_dir=data/word2vec
else
	echo "Invalid Option Selected"
fi
