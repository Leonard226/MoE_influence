from datasets import load_dataset
import torch

def c4_dataset_helper(dataset_len, min_words=64):
    """ Select some data from C4 dataset as test samples. """
    original_dataset = load_dataset(path="allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz") # len: 356318
    my_dataset = []
    counter = 0
    while len(my_dataset) < dataset_len:
        cur_text = original_dataset["train"][counter]["text"]
        if len(cur_text.split()) >= min_words: # to ensure that the prompt is not too short
            my_dataset.append(cur_text)
        counter += 1

    print("len of my_c4_dataset", len(my_dataset))
    del original_dataset
    return my_dataset