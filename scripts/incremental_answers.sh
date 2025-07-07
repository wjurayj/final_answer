DATA_DIR="./out"

## AIME R1
HF_TOKEN=$YOUR_HF_KEY conda run --no-capture-output -n s1 python scripts/incremental_answers.py \
    --model_name  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B\
    --data_dir "$DATA_DIR/aime/r1" \
    --folder_name "forcing8k" \
    --file_pattern "samples_aime24*" \
    --token_limit 8000 \
    --increment 100 \
    --think_delimiter "</think>" \
    --cache_file "./incremental/intermediate/probs_24_r1.jsonl" \
    --output_file "./incremental/final/results_24_r1.json"

HF_TOKEN=$YOUR_HF_KEY conda run --no-capture-output -n s1 python scripts/incremental_answers.py \
    --model_name  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B\
    --data_dir "$DATA_DIR/aime/r1" \
    --folder_name "forcing8k" \
    --file_pattern "samples_aime25*" \
    --token_limit 8000 \
    --increment 100 \
    --think_delimiter "</think>" \
    --cache_file "./incremental/intermediate/probs_25_r1.jsonl" \
    --output_file "./incremental/final/results_25_r1.json"


## AIME s1
HF_TOKEN=$YOUR_HF_KEY conda run --no-capture-output -n s1 python scripts/incremental_answers.py \
    --model_name "simplescaling/s1-32B" \
    --data_dir "$DATA_DIR/aime/s1" \
    --folder_name "forcing8k" \
    --file_pattern "samples_aime24*" \
    --token_limit 8000 \
    --increment 100 \
    --think_delimiter "<|im_start|>answer" \
    --cache_file "./incremental/intermediate/probs_24_s1.jsonl" \
    --output_file "./incremental/final/results_24_s1.json"

HF_TOKEN=$YOUR_HF_KEY conda run --no-capture-output -n s1 python scripts/incremental_answers.py \
    --model_name "simplescaling/s1-32B" \
    --data_dir "$DATA_DIR/aime/s1" \
    --folder_name "forcing8k" \
    --file_pattern "samples_aime25*" \
    --token_limit 8000 \
    --increment 100 \
    --think_delimiter "<|im_start|>answer" \
    --cache_file "./incremental/intermediate/probs_25_s1.jsonl" \
    --output_file "./incremental/final/results_25_s1.json"


## GPQA

HF_TOKEN=$YOUR_HF_KEY conda run --no-capture-output -n s1 python scripts/incremental_answers.py \
    --model_name  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B\
    --data_dir "$DATA_DIR/gpqa/r1" \
    --folder_name "forcing4k" \
    --file_pattern "samples_gpqa*" \
    --token_limit 4000 \
    --increment 50 \
    --think_delimiter "</think>" \
    --cache_file "./incremental/intermediate/probs_gpqa_r1.jsonl" \
    --output_file "./incremental/final/results_gpqa_r1.json" \
    --multiple_choice

HF_TOKEN=$YOUR_HF_KEY conda run --no-capture-output -n s1 python scripts/incremental_answers.py \
    --model_name "simplescaling/s1-32B" \
    --data_dir "$DATA_DIR/gpqa/s1" \
    --folder_name "forcing4k" \
    --file_pattern "samples_gpqa*" \
    --token_limit 4000 \
    --increment 50 \
    --think_delimiter "<|im_start|>answer" \
    --cache_file "./incremental/intermediate/probs_gpqa_s1.jsonl" \
    --output_file "./incremental/final/results_gpqa_s1.json" \
    --multiple_choice
