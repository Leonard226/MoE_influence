from datasets import load_dataset


def gsm8k_dataset_helper(dataset_len, min_words=64):
    """Select GSM8K grade-school math word problems. Concatenates the
    question and the chain-of-thought answer so each prompt has enough text
    to fill the 32-token build_dag window (questions alone are often short)."""
    original_dataset = load_dataset("openai/gsm8k", "main", split="train")
    my_dataset = []
    counter = 0
    while len(my_dataset) < dataset_len and counter < len(original_dataset):
        ex = original_dataset[counter]
        cur_text = ex["question"] + " " + ex["answer"]
        if len(cur_text.split()) >= min_words:
            my_dataset.append(cur_text)
        counter += 1

    print("len of my_gsm8k_dataset", len(my_dataset))
    del original_dataset
    return my_dataset