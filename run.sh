if [ "$1" = "vocab" ]; then
	python vocab.py data/msrvtt10k/TextData/msrvtt10ktrain.caption.txt vocab.json
elif [ "$1" = "tags" ]; then
	PYTHONPATH=. python scripts/video_tag.py data/msrvtt10k/TextData/msrvtt10ktrain.caption.txt \
	--tag_vocab_file=tag_vocab.json --annotation_file=video_tag.txt --vocab_size=512 --label_freq_cutoff=2
elif [ "$1" = "word2vec" ]; then
	PYTHONPATH=. python scripts/word2vec.py pretrained_weight.npy --vocab_file=vocab.json --word2vec_dir=data/word2vec
elif [ "$1" = "tiny" ]; then
	train_cap_basename="data/msrvtt10k/TextData/msrvtt10ktrain.caption"
	train_cap_file="${train_cap_basename}.txt"
	train_cap_tiny_file="${train_cap_basename}_tiny.txt"
	val_cap_file="data/msrvtt10k/TextData/msrvtt10kval.caption.txt"
	test_cap_file="data/msrvtt10k/TextData/msrvtt10ktest.caption.txt"

	PYTHONPATH=. python scripts/create_tiny.py --train_cap_file=$train_cap_file --val_cap_file=$val_cap_file --test_cap_file=$test_cap_file
	python vocab.py $train_cap_tiny_file vocab_tiny.json --freq_cutoff=1
	PYTHONPATH=. python scripts/video_tag.py $train_cap_tiny_file --tag_vocab_file=tag_vocab_tiny.json --annotation_file=video_tag_tiny.txt \
	--vocab_freq_cutoff=1 --label_freq_cutoff=1
else
	echo "Invalid Option Selected"
fi
