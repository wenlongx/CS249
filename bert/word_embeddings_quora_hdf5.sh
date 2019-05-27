export BERT_BASE_DIR="/home/max/Documents/NLP-playground/uncased_L-12_H-768_A-12"
export QUORA_INPUT="/home/max/Documents/CS249/quora-insincere-questions-classification/1000_example_train_doc.txt"
# export QUORA_INPUT="/home/max/Documents/NLP-playground/bert-word-embeddings/word_embeddings_quora_input.txt"

python bert/extract_features_hdf5.py \
  --input_file=$QUORA_INPUT \
  --output_file_word=bert-word-embeddings/word_embeddings_quora_output_word.hdf5 \
  --output_file_sentence=bert-word-embeddings/word_embeddings_quora_output_sentence.hdf5 \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=128 \
  --batch_size=8
