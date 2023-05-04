if [ "$1" = "vocab" ]; then
	python vocab.py data/msrvtt10k/TextData/msrvtt10ktrain.caption.txt vocab.json

elif [ "$1" = "tags" ]; then
	PYTHONPATH=. python scripts/video_tag.py data/msrvtt10k/TextData/msrvtt10ktrain.caption.txt \
	--tag_vocab_file=tag_vocab.json --annotation_file=video_tag.txt --vocab_size=512 --label_freq_cutoff=2

elif [ "$1" = "word2vec" ]; then
	PYTHONPATH=. python scripts/word2vec.py pretrained_weight.npy --vocab_file=vocab.json --word2vec_dir=data/word2vec

elif [ "$1" = "tiny" ]; then
	tiny_cap_file="data/msrvtt10k/TextData/msrvtt10k.caption_tiny.txt"
	tiny_cap_num=${2:-"10"}

	PYTHONPATH=. python scripts/caption_tiny.py --train_cap_file=data/msrvtt10k/TextData/msrvtt10ktrain.caption.txt \
	--tiny_cap_file=$tiny_cap_file --cap_num=$tiny_cap_num
	python vocab.py $tiny_cap_file vocab_tiny.json --freq_cutoff=1
	PYTHONPATH=. python scripts/video_tag.py $tiny_cap_file --tag_vocab_file=tag_vocab_tiny.json --annotation_file=video_tag_tiny.txt \
	--vocab_freq_cutoff=1 --label_freq_cutoff=1

elif [ "$1" = "train_hybrid_cpu" ]; then
	python run_hybrid.py train config_hybrid.yaml --device=cpu

elif [ "$1" = "train_hybrid_cuda" ]; then
	CUDA_VISIBLE_DEVICES=0 python run_hybrid.py train config_hybrid.yaml --device=cuda

elif [ "$1" = "train_hybrid_mps" ]; then
	python run_hybrid.py train config_hybrid.yaml --device=mps

elif [ "$1" = "train_hybrid_cpu_debug" ]; then
	python run_hybrid.py train config_hybrid.yaml --device=cpu --debug

elif [ "$1" = "train_hybrid_mps_debug" ]; then
	python run_hybrid.py train config_hybrid.yaml --device=mps --debug

elif [ "$1" = "test_hybrid_cpu" ]; then
	if [ ! -z "$2" ]; then
		python run_hybrid.py test config_hybrid.yaml --device=cpu --model_path=$2
	else
		echo "second arg. MODEL_PATH not passed"
	fi

elif [ "$1" = "test_hybrid_cuda" ]; then
	if [ ! -z "$2" ]; then
		CUDA_VISIBLE_DEVICES=0 python run_hybrid.py test config_hybrid.yaml --device=cuda --model_path=$2
	else
		echo "second arg. MODEL_PATH not passed"
	fi

elif [ "$1" = "test_hybrid_mps" ]; then
	python run_hybrid.py test config_hybrid.yaml --device=mps
	if [ ! -z "$2" ]; then
		python run_hybrid.py test config_hybrid.yaml --device=mps --model_path=$2
	else
		echo "second arg. MODEL_PATH not passed"
	fi

elif [ "$1" = "test_hybrid_cpu_debug" ]; then
	if [ ! -z "$2" ]; then
		python run_hybrid.py test config_hybrid.yaml --device=cpu --debug --model_path=$2
	else
		echo "second arg. MODEL_PATH not passed"
	fi

elif [ "$1" = "test_hybrid_mps_debug" ]; then
	if [ ! -z "$2" ]; then
		python run_hybrid.py test config_hybrid.yaml --device=mps --debug --model_path=$2
	else
		echo "second arg. MODEL_PATH not passed"
	fi

else
	echo "Invalid Option Selected"
fi
