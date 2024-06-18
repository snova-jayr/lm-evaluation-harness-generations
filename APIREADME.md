# How to reproduce our results?

This repository is identical to the original LM-Eval-Harness from EleutherAI with two major changes. 
1. We implement both Groq and SambaNova API's here.
2. We turn many multiple-choice tasks into generative tasks such as HellaSwag.

## List of tasks

Here are the tasks we test on. Next to each name is the name of the task registered by LM-Eval-Harness. If you want more tasks to run, feel free to reach out to etash.guha@sambanovasystems.com on how to implement them. 

1. MMLU STEM: "mmlu_stem_generative"
2. MMLU Other: "mmlu_other_generative" 
3. MMLU Social Sciences: "mmlu_social_sciences_generative"
4. MMLU Humanities: "mmlu_humanities_generative"
5. CoQA: "coqa"
6. AGI_EvalMath: "agieval_math"
7. BigBenchHardFewshot: "bbh_fewshot"
8. ARC-Easy*: "arc_easy_generative"
9. ARC-Challenge*: "arc_challenge_generative"
10. OpenBookQA*: "openbookqa_generative"
11. HellaSwag*: "hellaswag_generative"
12. PiQA*: "piqa_generative"
13. SciQ*: "sciq_generative"
14. AExams*: "aexams_generative"

Here are some tasks you can run for example
1. TruthfulQA: "truthfulqa_gen"
2. AGIEval: "agieval"
3. EQBench: "eq_bench"


## Groq

```
bash_script_templates/run_groq_agi_bigbench_tasks.sh
```

Here, change the key to your Groq key and the task name to whatever task you want to evaluate.

## SambaNova

```
bash_script_templates/run_sambanova_agi_bigbench_tasks.sh
```

Here, change the key to your Samba key and the task name to whatever task you want to evaluate.


# Changes we made

* Modified some tasks so that we would be able to do generation based comparison rather than prompt logit comparison
* Modified `lm_eval/api/samplers.py` and `lm_eval/api/task.py` to handle chat history based formatting for optimal results. We saw a significant increase in accuracy on gsm8k_cot for both sambanova and groq. This may be due to the proper format that Llama 3 8B requires. In turn also had to modify `lm_eval/evaluator.py` for the hashing of the prompts due to changing from strings to list
* Created models for both sambanova (`lm_eval/models/sambanova.py`) and groq (`lm_eval/models/groq.py`)
