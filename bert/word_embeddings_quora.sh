export BERT_BASE_DIR="/home/max/Documents/NLP-playground/uncased_L-12_H-768_A-12"

python bert/extract_features.py \
  --input_file=word_embeddings_quora_input.txt \
  --output_file=word_embeddings_quora_output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
