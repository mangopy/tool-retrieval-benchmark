# Setting environment variables
export CUDA_VISIBLE_DEVICES="1,2,3,4"
export TOKENIZERS_PARALLELISM=False

# Defining Variables
MODEL_NAME="intfloat/e5-small-v2"
CACHE_DIR="./train/cache/model"
TRAIN_DATA="./data/train_data/train.jsonl"
CACHE_PATH="./train/cache/data"
OUTPUT_DIR="./train/e5-small-v2-train-checkpoint"
DEEPSPEED_CONFIG="./train/ds_stage0.json"
QUERY_INSTRUCTION_FOR_RETRIEVAL="Represent this sentence for searching relevant passages: "
PER_DEVICE_TRAIN_BATCH_SIZE=4

# Running the training with the specified parameters
torchrun --master_port 29400 --nproc_per_node 4 \
	-m FlagEmbedding.finetune.embedder.encoder_only.base \
	--model_name_or_path $MODEL_NAME \
    --cache_dir $CACHE_DIR \
    --train_data $TRAIN_DATA \
    --cache_path $CACHE_PATH \
	--output_dir $OUTPUT_DIR \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval "$QUERY_INSTRUCTION_FOR_RETRIEVAL" \
    --query_instruction_format "{}{}" \
    --knowledge_distillation False \
    --overwrite_output_dir \
    --learning_rate 1e-6 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed $DEEPSPEED_CONFIG \
    --logging_steps 1000 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div
