export BERT_BASE_DIR="/home/max/Documents/NLP-playground/uncased_L-12_H-768_A-12"
export QUORA_DIR="/home/max/Documents/NLP-playground/quora-insincere"

python bert/run_classifier_quora.py \
  --task_name=quora \
  --do_train=true \
  --do_eval=true \
  --data_dir=$QUORA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=32 \
  --train_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$QUORA_DIR/bert_output
