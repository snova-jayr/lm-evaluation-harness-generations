

def doc_to_text(doc):
    prompt = f"Question: {doc['goal']}\n1. {doc['sol1']}\n 2.\{doc['sol2']}\nAnswer:"
    return prompt

def doc_to_target(doc):
    return str(doc['label'] + 1)