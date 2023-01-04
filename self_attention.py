
import torch


def basic_self_attention(X):
    """
    No learnable parameters
    """
    # compute dot product with itself to generate attention weights
    attention_weights = torch.matmul(X, X.T)
    print(attention_weights.shape)

    # normalize using softmax
    attention_weights = torch.softmax(attention_weights, dim=1)
    print(attention_weights)

    # multiply attention weights with input to get weighted sum
    weighted_sum = torch.matmul(attention_weights, X)
    print(weighted_sum.shape)

    return weighted_sum


def self_attention(Q, K, V, scaled=False):
    """
    Learnable parameters
    """
    if scaled:
        # scale the dot product by the square root of the dimensionality
        Q = Q / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))

    # compute dot product with itself to generate attention weights
    attention_weights = torch.matmul(Q, K.T)

    # normalize using softmax
    attention_weights = torch.softmax(attention_weights, dim=1)

    # multiply attention weights with input to get weighted sum
    weighted_sum = torch.matmul(attention_weights, V)

    return weighted_sum


# pre-amble
embedding_size = 4
input_seq_size = 512
num_heads = 8

X = torch.randn(input_seq_size, embedding_size)
Q = torch.randn(input_seq_size, embedding_size * num_heads)
K = torch.randn(input_seq_size, embedding_size * num_heads)
V = torch.randn(input_seq_size, embedding_size * num_heads)
W_o = torch.randn(embedding_size * num_heads, embedding_size)

# shape = (input_seq_size, embedding_size)
weighted_sum = self_attention(Q, K, V, scaled=True)
print(weighted_sum.shape)

# shape = (input_seq_size, embedding_size)
output = torch.matmul(weighted_sum, W_o)
print(output.shape)











