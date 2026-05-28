from datasets import load_dataset


def wikitext2_dataset_helper(dataset_len, min_words=64):
    """Select WikiText-2 (raw) samples as test samples."""
    original_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    my_dataset = []
    counter = 0
    while len(my_dataset) < dataset_len and counter < len(original_dataset):
        cur_text = original_dataset[counter]["text"]
        if len(cur_text.split()) >= min_words:
            my_dataset.append(cur_text)
        counter += 1

    print("len of my_wikitext2_dataset", len(my_dataset))
    del original_dataset
    return my_dataset
