echo $SLURM_GPUS

YOUR_HF_KEY="<YOUR_HF_KEY>"

YOUR_OPENAI_KEY=token-abc123
PARALLELISM=$SLURM_GPUS
PROCESSOR_MODEL="meta-llama/Llama-3.1-70B-Instruct"

OUT_DIR=out/aime

MODEL_ALIAS="s1"

if [ "$MODEL_ALIAS" = "s1" ]; then
    MODEL_NAME="simplescaling/s1-32B"
elif [ "$MODEL_ALIAS" = "r1" ]; then
    MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
else
    echo "Invalid MODEL_ALIAS: $MODEL_ALIAS"
    exit 1
fi


OUT_PATH=${OUT_DIR}/${MODEL_ALIAS}

#for camera ready
OPENAI_API_KEY=$YOUR_OPENAI_KEY PROCESSOR=$PROCESSOR_MODEL HF_TOKEN=$YOUR_HF_KEY conda run --no-capture-output -n s1 lm_eval --model vllm --model_args pretrained=$MODEL_NAME,tokenizer=Qwen/Qwen2.5-32B-Instruct,dtype=float32,tensor_dparallel_size=$PARALLELISM,enable_chunked_prefill=False,enable_prefix_caching=False,max_model_len=32768 --tasks aime25_nofigures,aime24_nofigures --batch_size auto --apply_chat_template --output_path ${OUT_PATH}/forcing8k --log_samples --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=8000,thinking_start=<|im_start|>,thinking_end=<|im_start|>answer,thinking_n_ignore=12,thinking_n_ignore_str=Wait"


MODEL_ALIAS="r1"

if [ "$MODEL_ALIAS" = "s1" ]; then
    MODEL_NAME="simplescaling/s1-32B"
elif [ "$MODEL_ALIAS" = "r1" ]; then
    MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
else
    echo "Invalid MODEL_ALIAS: $MODEL_ALIAS"
    exit 1
fi


OUT_PATH=${OUT_DIR}/${MODEL_ALIAS}


# for camera ready
OPENAI_API_KEY=$YOUR_OPENAI_KEY PROCESSOR=$PROCESSOR_MODEL HF_TOKEN=$YOUR_HF_KEY conda run --no-capture-output -n s1 lm_eval --model vllm --model_args pretrained=$MODEL_NAME,tokenizer=Qwen/Qwen2.5-32B-Instruct,dtype=float32,tensor_parallel_size=$PARALLELISM,enable_chunked_prefill=False,enable_prefix_caching=False,max_model_len=44490 --tasks aime25_nofigures,aime24_nofigures --batch_size auto --apply_chat_template --output_path ${OUT_PATH}/forcing8k --log_samples --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=8000,thinking_start=<｜Assistant｜><think>,thinking_end=</think>,thinking_n_ignore=12,thinking_n_ignore_str=Wait"


OUT_DIR=out/gpqa

MODEL_ALIAS="s1"

if [ "$MODEL_ALIAS" = "s1" ]; then
    MODEL_NAME="simplescaling/s1-32B"
elif [ "$MODEL_ALIAS" = "r1" ]; then
    MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
else
    echo "Invalid MODEL_ALIAS: $MODEL_ALIAS"
    exit 1
fi


OUT_PATH=${OUT_DIR}/${MODEL_ALIAS}

# # ## WITH **Final Answer**
OPENAI_API_KEY=$YOUR_OPENAI_KEY PROCESSOR=$PROCESSOR_MODEL HF_TOKEN=$YOUR_HF_KEY conda run --no-capture-output -n s1 lm_eval --model vllm --model_args pretrained=$MODEL_NAME,tokenizer=Qwen/Qwen2.5-32B-Instruct,dtype=float32,tensor_parallel_size=$PARALLELISM,enable_chunked_prefill=False,enable_prefix_caching=False,max_model_len=32768 --tasks gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path ${OUT_PATH}/forcing4k --log_samples --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=4000,temperature=0,thinking_start=<|im_start|>,thinking_end=<|im_start|>answer,thinking_n_ignore=10,thinking_n_ignore_str=Wait"



MODEL_ALIAS="r1"

if [ "$MODEL_ALIAS" = "s1" ]; then
    MODEL_NAME="simplescaling/s1-32B"
elif [ "$MODEL_ALIAS" = "r1" ]; then
    MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
else
    echo "Invalid MODEL_ALIAS: $MODEL_ALIAS"
    exit 1
fi


OUT_PATH=${OUT_DIR}/${MODEL_ALIAS}


## WITH </think> TAG, HELPS WITH R1
OPENAI_API_KEY=$YOUR_OPENAI_KEY PROCESSOR=$PROCESSOR_MODEL HF_TOKEN=$YOUR_HF_KEY conda run --no-capture-output -n s1 lm_eval --model vllm --model_args pretrained=$MODEL_NAME,tokenizer=Qwen/Qwen2.5-32B-Instruct,dtype=float32,tensor_parallel_size=$PARALLELISM,enable_chunked_prefill=False,enable_prefix_caching=False,max_model_len=44490 --tasks gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path ${OUT_PATH}/forcing4k --log_samples --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=4000,temperature=0,thinking_start=<｜Assistant｜><think>,thinking_end=</think>,thinking_n_ignore=10,thinking_n_ignore_str=Wait"


