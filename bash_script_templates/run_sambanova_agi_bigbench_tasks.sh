task_name=bigbench_generate_until

SAMBA_URL=<INSERTSAMBAURL> \
SAMBA_KEY=<INSERTSAMBAKEY> \
lm_eval \
    --model snsdk \
    --tasks ${task_name} \
    --write_out \
    --output_path sambanova_results/${task_name}.jsonl \
    --log_samples \
    --batch_size 1
