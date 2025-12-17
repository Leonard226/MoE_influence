import torch
from tqdm import tqdm
from tools.misc import project_to_logits
def entropy_from_logits(logits, dim=-1):
    log_p = torch.log_softmax(logits, dim)
    p = torch.softmax(logits, dim)
    return -(p * log_p).sum(dim)


def find_entropy(prompt_ls, model, tokenizer, router_weight_ls, max_token_per_prompt, bsz=100):
    batch_token = tokenizer(prompt_ls, return_tensors="pt", max_length=max_token_per_prompt, padding=False, truncation=True)
    n_prompts, max_n_tokens = batch_token["attention_mask"].shape
    router_weight_vectors = torch.stack(router_weight_ls, dim=0) # shape: [n_layers, n_experts, n_dim]
    for B in tqdm(range(0, n_prompts, bsz)):
        model_outputs, hook_dict = model(input_ids=batch_token["input_ids"][B:B+bsz], attention_mask=batch_token["attention_mask"][B:B+bsz])
        after_norm2 = hook_dict["hook_after_norm2"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        prediction = model_outputs[0] # [batch_size, n_tokens, vocab_size]
        print(prediction.shape)
        print(entropy_from_logits(prediction).shape)
        print('pred', entropy_from_logits(prediction)[0,:])
        original_score = torch.einsum("RED,PRTD->PTER", router_weight_vectors, after_norm2)
        top_n_scores, _ = torch.sort(original_score, dim=2, descending=True)
        print('top k entropy', entropy_from_logits(top_n_scores[0,:,:8,-1], dim=1))

        print('all experts entropy',entropy_from_logits(original_score, dim=2)[0,:,-1])
        # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        final_var = hook_dict["hook_layer_output"][0, -1, :, :].pow(2).mean(-1, keepdim=True)
        tmp_logits = torch.zeros((14, 50304))
        for k in range(14):
            tmp_logits[k] = project_to_logits(after_norm2[0,-1,k], final_var[k], model)
        print(entropy_from_logits(tmp_logits, dim=1))
        exit()
        
    exit()