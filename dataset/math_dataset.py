from datasets import load_dataset
import torch

def open_r1_math_dataset_helper(dataset_len, seed=None, min_words=64):
    """ Select some data from OpenR1-Math dataset as test samples. """
    original_dataset = load_dataset(path="open-r1/OpenR1-Math-220k")
    my_dataset = []
    counter = 0
    if seed == None: # sequentially select samples
        while len(my_dataset) < dataset_len:
            cur_text = original_dataset["train"][counter]["problem"]
            if len(cur_text.split()) >= min_words: # to ensure that the prompt is not too short
                my_dataset.append(cur_text)
            counter += 1
    else:
        torch.manual_seed(seed)
        text_id_ls = torch.randperm(len(original_dataset["train"]))
        while len(my_dataset) < dataset_len:
            cur_text = original_dataset["train"][text_id_ls[counter].item()]["problem"]
            if len(cur_text.split()) >= min_words: # to ensure that the prompt is not too short
                my_dataset.append(cur_text)
            counter += 1
    print("len of my_open_r1_math_dataset", len(my_dataset))
    del original_dataset
    return my_dataset

open_r1_math_dataset_helper(100, min_words=40)
open_r1_math_dataset_helper(100, seed=20, min_words=40)