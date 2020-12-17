BERT_BASE_DIR="/home/stevens.994/language/data/assets/pretrained_bert/uncased_L-12_H-768_A-12"
BERT_LARGE_DIR="/home/stevens.994/language/data/assets/pretrained_bert/wwm_uncased_L-24_H-1024_A-16"
GLUE_DIR="glue_data"

BERT_DIR=$BERT_LARGE_DIR

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=2 \
  --eval_batch_size=2 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/home/stevens.994/mrpc_output/ \
  --save_checkpoints_steps=1000 \

