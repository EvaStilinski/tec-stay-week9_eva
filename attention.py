import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value):
    """
    Calculate the attention weights.
    query, key, value must have matching leading dimensions.
    The sequence to be attended to (key, value) must be broadcastable to the query sequences shape.
    """
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))  # Dot product of query and key

    # Scale matmul_qk
    depth = key.shape[-1]
    scaled_attention_logits = matmul_qk / (depth ** 0.5)

    # Apply softmax to get probabilities
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, value)  # Multiply by values
    return output, attention_weights

# Example usage
def example():
    # Random tensors representing query, key, and value
    tmp_query = torch.rand(1, 60, 512)  # (batch_size, seq_length, depth)
    tmp_key = torch.rand(1, 60, 512)
    tmp_value = torch.rand(1, 60, 512)

    # Apply the attention mechanism
    output, attention_weights = scaled_dot_product_attention(tmp_query, tmp_key, tmp_value)

    print("Output:", output.size())
    print("Attention Weights:", attention_weights.size())

example()
