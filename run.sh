if [ "$1" = "vocab" ]; then
	python vocab.py data/msrvtt10k/TextData/msrvtt10ktrain.caption.txt vocab.json
elif [ "$1" = "tags" ]; then
	PYTHONPATH=. python scripts/video_tag.py data/msrvtt10k/TextData/msrvtt10ktrain.caption.txt \
	--tag_vocab_file=tag_vocab.json --annotation_file=video_tag.txt --vocab_size=512 --label_freq_cutoff=2
elif [ "$1" = "word2vec" ]; then
	PYTHONPATH=. python scripts/word2vec.py pretrained_weight.npy --vocab_file=vocab.json --word2vec_dir=data/word2vec
else
	echo "Invalid Option Selected"
fi
