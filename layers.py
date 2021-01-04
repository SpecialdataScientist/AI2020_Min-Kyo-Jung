import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scaled_term, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scaled_term = scaled_term
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # Score
        attn_score = torch.matmul(q, k)  # [B, n_head, T, T]    # attention score
        attn_score = attn_score / self.scaled_term

        # because max_size, blank is 0
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e10)

        # attention distribution
        attn_prob = self.softmax(attn_score)  # [B, n_head, T, T]
        attn_prob = self.dropout(attn_prob)

        # attention value
        output = torch.matmul(attn_prob, v)  # [B, n_head, T, H//n_head]

        # return attention value, attention score
        return output, attn_score


class MultiHeadAttention(nn.Module):
    # n_head is parallel num
    def __init__(self, input_size, hidden_size, n_head, device, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head

        # weight matrix for Q, K V matrix
        # weight matrix hidden_size size eauql q, k ,v matrix ouput size
        self.w_q = nn.Linear(input_size, hidden_size)   # shape [input_size, hidden_size]
        self.w_k = nn.Linear(input_size, hidden_size)
        self.w_v = nn.Linear(input_size, hidden_size)
        self.scaled_dot_attention = ScaledDotProductAttention(
            torch.sqrt(torch.FloatTensor([input_size // n_head])).to(device))   # input_size / num_head == k dimension
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, hidden_size)     # input_size == output_size

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        '''
        query = key = value: [B, T, H]
        mask: [B, T, 1]
        '''
        # Project and split
        q = self.w_q(q).view(batch_size, -1, self.n_head,
                             self.hidden_size // self.n_head)  # [B, T, H] -> [B, T, n_head, H//n_head]
        k = self.w_k(k).view(batch_size, -1, self.n_head,
                             self.hidden_size // self.n_head)  # [B, T, H] -> [B, T, n_head, H//n_head]
        v = self.w_v(v).view(batch_size, -1, self.n_head,
                             self.hidden_size // self.n_head)  # [B, T, H] -> [B, T, n_head, H//n_head]

        q = q.permute(0, 2, 1, 3)  # [B, n_head, T, H//n_head]
        k = k.permute(0, 2, 3, 1)  # [B, n_head, H//n_head, T]
        v = v.permute(0, 2, 1, 3)  # [B, n_head, T, H//n_head]

        output, attn = self.scaled_dot_attention(q, k, v, mask)  # [B, n_head, T, T], [B, n_head, T, H//n_head]

        output = output.transpose(1, 2).contiguous()  # [B, T, n_head, H//n_head]
        output = output.view(batch_size, -1, self.n_head * (self.hidden_size // self.n_head))  # [B, T, H]

        output = self.output_layer(output)

        output = self.dropout(output)

        # return output, attention score
        return output, attn