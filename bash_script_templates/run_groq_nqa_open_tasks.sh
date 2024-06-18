TASK=nq_open
GROQ_API_KEY=<INSERTGROQKEY> lm_eval --model groq --tasks $TASK --write_out --output_path groq_results/$TASK --log_samples --batch_size 1 --num_fewshot 0
