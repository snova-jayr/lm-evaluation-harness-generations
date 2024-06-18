

def doc_to_text(doc):
    prompt = f"Question: {doc['question']}"
    for i in range(len(doc['choices']['text'])):
        prompt += f"\n{doc['choices']['label'][i]}. {doc['choices']['text'][i]}"
    prompt += "\nAnswer:"
    return prompt