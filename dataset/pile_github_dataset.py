from datasets import load_dataset


def pile_github_dataset_helper(dataset_len, min_words=64):
    """Select Github (multi-language code) samples from the Pile-uncopyrighted
    dataset.

    Streams the Pile and filters by `meta.pile_set_name == "Github"`. Github
    is one of the larger Pile subsets (~8% of the corpus), so reaching
    `dataset_len` requires iterating through roughly 12x that many records.
    Complements `codeparrot-clean` (Python-only) by covering multiple
    programming languages.
    """
    original_dataset = load_dataset(
        "monology/pile-uncopyrighted", split="train", streaming=True
    )
    my_dataset = []
    for ex in original_dataset:
        if ex.get("meta", {}).get("pile_set_name") != "Github":
            continue
        cur_text = ex["text"]
        if len(cur_text.split()) >= min_words:
            my_dataset.append(cur_text)
            if len(my_dataset) >= dataset_len:
                break

    print("len of my_pile_github_dataset", len(my_dataset))
    return my_dataset