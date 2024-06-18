task_name=bigbench_generate_until

GROQ_API_KEY=<INSERTGROQKEY> \
lm_eval \
    --model groq \
    --tasks ${task_name} \
    --write_out \
    --output_path groq_results/${task_name}.jsonl \
    --log_samples \
    --batch_size 1