from datasets import load_dataset
import torch

def open_r1_math_dataset_helper(dataset_len, seed=None, min_words=64):
    """ Select some data from OpenR1-Math dataset as test samples. """
    original_dataset = load_dataset(path="open-r1/OpenR1-Math-220k")
    my_dataset = []
    counter = 0
    while len(my_dataset) < dataset_len:
        cur_text = original_dataset["train"][counter]["problem"]
        if len(cur_text.split()) >= min_words: # to ensure that the prompt is not too short
            my_dataset.append(cur_text)
        counter += 1
    print("len of my_open_r1_math_dataset", len(my_dataset))
    del original_dataset
    return my_dataset