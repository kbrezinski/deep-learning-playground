
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


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_e, num_heads, scaled=False):
        super().__init__()
        self.d_e = d_e
        self.scaled = scaled
        self.num_heads = num_heads

        self.W_q = torch.nn.Parameter(torch.randn(d_e, d_e * num_heads))
        self.W_k = torch.nn.Parameter(torch.randn(d_e, d_e * num_heads))
        self.W_v = torch.nn.Parameter(torch.randn(d_e, d_e * num_heads))
        self.W_o = torch.nn.Parameter(torch.randn(d_e, d_e * num_heads))

    def __call__(self, X):

        Q = torch.matmul(X, self.W_q)
        K = torch.matmul(X, self.W_k)
        V = torch.matmul(X, self.W_v)

        if self.scaled:
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
T = 10
d_e = 512
num_heads = 2

# embedding of word sentece
X = torch.randn(T, d_e)

attention = ScaledDotProductAttention(d_e, scaled=True)
weighted_sum = attention(X)
print(weighted_sum.shape)

# shape = (input_seq_size, embedding_size)
#output = torch.matmul(weighted_sum, W_o)
#print(output.shape)











