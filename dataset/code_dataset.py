from datasets import load_dataset
import torch


def code_dataset_helper(dataset_len, seed=None, min_words=64):
    """Select Python code samples from codeparrot-clean as test samples.

    Mirrors the interface of c4_dataset_helper / open_r1_math_dataset_helper.
    Uses streaming to avoid downloading the full ~50GB corpus; we iterate
    until we have collected `dataset_len` samples that pass the min_words
    filter.

    Args:
        dataset_len: how many code samples to return.
        seed: if None, take the first `dataset_len` samples sequentially from
            the stream. If an int, shuffle the stream with this seed first.
        min_words: drop samples whose `content.split()` length is below this.
    """
    ds = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True,
    )

    if seed is not None:
        torch.manual_seed(seed)
        # buffer_size controls the randomness window for streaming shuffle.
        ds = ds.shuffle(seed=seed, buffer_size=10000)

    my_dataset = []
    for sample in ds:
        if len(my_dataset) >= dataset_len:
            break
        cur_text = sample["content"]
        if len(cur_text.split()) >= min_words:
            my_dataset.append(cur_text)

    print(f"len of my_code_dataset {len(my_dataset)}")
    return my_dataset
