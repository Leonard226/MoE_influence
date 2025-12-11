## NOTE: The following code was not used in the experiments in the paper but is provided for reference.

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def layer_print(model):
    """ Print layer info of the assigned model. """
    for k, v in model.state_dict().items():
        print(k, v.shape)
    print(model.config)

def run_template(prompt_ls, model, tokenizer):
    """ A simple template to show how to obtain some basic info. """
    batch_token = tokenizer(prompt_ls, return_tensors="pt", padding=True)
    model_outputs, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])

    ## Code for checking
    print("prompts:", tokenizer.batch_decode(batch_token["input_ids"]))
    for bt in batch_token["input_ids"]:
        print("token_id:", bt)
        print("decode:", [tokenizer.decode(x) for x in bt])
    prediction = model_outputs[0] # [batch_size, n_tokens, vocab_size]
    predicted_top10 = torch.argsort(prediction[0, -1], descending=True)[:10] # 0=first prompt, -1=last token
    predicted_text = [tokenizer.decode(x) for x in predicted_top10]
    print("top10 predicted_text of the first prompt at the last token:", predicted_text)

    return batch_token, model_outputs, hook_dict

def matrix_drawer(data, name, output_dir, cmap_set="RdBu", title="", xlabel="", ylabel=""):
    """ A simple template for visualizing a matrix. """
    data = data.detach().cpu().numpy() # 2-D data
    plt.figure(figsize=(11,11))
    plt.imshow(data, cmap=cmap_set)
    for r in range(data.shape[1]):
        for c in range(data.shape[0]):
            plt.text(r, c, np.round(data[c, r], 2), fontsize=10, horizontalalignment="center", verticalalignment="center")
    plt.colorbar()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_dir + name + ".png")
    plt.close("all")

def scatter_drawer(data, name, output_dir, title="", xlabel="", ylabel=""):
    """ A simple template for visualizing a scatter plot. """
    data = data.detach().cpu().numpy() # 2-D data
    n_dim1, n_dim2 = data.shape
    xs = [i for i in range(n_dim1) for _ in range(n_dim2)]
    ys = data.reshape(-1)
    plt.scatter(xs, ys, alpha=0.3, s=5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_dir + name + ".png")
    plt.close("all")

def cosine_similarity(inputs_vectors, router_weight_vectors):
    """ Compute the cosine similarity between the vectors (for function 'decompose_XA_verbose')
    :param1 inputs_vectors | shape:[num1, n_dim]
    :param2 router_weight_vectors | shape:[num2, n_dim]
    :return: a result matrix
    """
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos_sim(inputs_vectors, router_weight_vectors)