from datasets import load_dataset


def humaneval_dataset_helper(dataset_len, min_words=8):
    """Select HumanEval prompts (function signatures + docstrings) as test
    samples. The dataset has only 164 entries total; if `dataset_len` is
    larger than that, this helper returns whatever passes the `min_words`
    filter. We keep `min_words` low because some HumanEval prompts are short.
    """
    original_dataset = load_dataset("openai/openai_humaneval", split="test")
    my_dataset = []
    for ex in original_dataset:
        cur_text = ex["prompt"]
        if len(cur_text.split()) >= min_words:
            my_dataset.append(cur_text)
            if len(my_dataset) >= dataset_len:
                break

    print("len of my_humaneval_dataset", len(my_dataset))
    del original_dataset
    return my_dataset