

def doc_to_text(doc):
    id = abs(hash(doc['question'])) % 4

    labels = list(range(4))
    answers = ['distractor1', 'distractor2', 'distractor3']
    answers.insert(id, "correct_answer")
    prompt = f"Question: {doc['question']}"
    for i in range(4):
        prompt += f"\n{labels[i]}. {doc[answers[i]]}"
    prompt += "\nAnswer:"
    return prompt


def doc_to_target(doc):
    id = abs(hash(doc['question'])) % 4
    return str(id)