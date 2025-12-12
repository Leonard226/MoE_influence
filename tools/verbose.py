import torch

def rmsnorm_breakdown(vector, components, layer_id, model, variance_epsilon=1e-05):
    """ Apply RMSNorm on the components (Single prompt)
    :param1 vector: the input of a RMSNorm
    :param2 components: components of the "vector". Note that their sum should be equal to the "vector".
    :param3 layer_id: which layer? Should be an integer.
    :param4 model: assigned model
    :param5 variance_epsilon: term for numerical stability
    :return: a list of RMSNorm-ed components
    """
    variance = vector.pow(2).mean(-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + variance_epsilon)
    weight = model.model.layers[layer_id].post_attention_layernorm.weight.data
    breakdowns = [weight * (i * rsqrt) for i in components]
    return breakdowns