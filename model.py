"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    token_2_id = {special_token: idx for idx, special_token in enumerate(specials)}
    idx = len(specials)
    for text in sentences:
        for word in text.split():
            if word not in token_2_id:
                token_2_id[word] = idx
                idx += 1
    return token_2_id

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    return {idx: token for token, idx in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    ids_lst = []
    for token in sentence.split():
        idx = token if token in token_to_id else unk_token
        ids_lst.append(token_to_id[idx])
    return ids_lst

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    token_list = []
    for idx in ids:
        token_list.append(id_to_token[idx])
    return token_list

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id for _ in range(max_len - len(ids))]

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    scalor = math.sqrt(d_model)
    return embeddings * scalor

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    return torch.tensor(10000 ** (-2 * torch.arange(0, d_model // 2) / d_model), dtype=torch.float)

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    return torch.arange(0, max_len, dtype=torch.float).reshape(-1, 1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    pe = pe.clone()
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe = pe.clone()
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros(max_len, d_model)
    position = build_position_index_column(max_len)
    div = compute_positional_div_term(d_model)
    pe = fill_even_indices_with_sin(pe, position, div)
    pe = fill_odd_indices_with_cos(pe, position, div)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    max_len = min(positional_encoding.shape[0], embedded_batch.shape[1])
    return embedded_batch + positional_encoding[:max_len]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    shape = token_ids.shape
    return (token_ids != pad_id).reshape(shape[0], 1, 1, shape[1])

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    bool_ones = torch.ones(seq_len, seq_len, dtype=torch.bool)
    causal_mask = torch.triu(bool_ones).T.reshape(1, 1, *bool_ones.shape)
    return causal_mask

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return torch.logical_and(padding_mask, causal_mask)

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    key_t = torch.transpose(key, -2, -1)
    return torch.matmul(query, key_t)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(~mask, float("-inf"))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    softmax_scores = torch.nn.functional.softmax(masked_scores, dim=-1)
    return torch.nan_to_num(softmax_scores, nan=0.0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    d_k = query.shape[-1]
    scores = compute_raw_attention_scores(query, key)
    scores = scale_attention_scores(scores, d_k)
    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores, mask)
    attention_weights = softmax_attention_weights(scores)
    attention = apply_attention_weights_to_values(attention_weights, value)
    return attention, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    d_k = tensor.shape[-1] // num_heads
    tensor = tensor.view(*tensor.shape[:-1], num_heads, d_k)
    return tensor

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    multi_head_tensor = multi_head_tensor.transpose(1, 2)
    shape = multi_head_tensor.shape
    d_h = shape[-1] * shape[-2]
    multi_head_tensor = multi_head_tensor.reshape(*shape[:-2], d_h)
    return multi_head_tensor

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    linear_projection = x @ weight.T
    if bias is not None:
        linear_projection += bias
    return linear_projection

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    Q = apply_linear_projection(x, w_q, b_q)
    K = apply_linear_projection(x, w_k, b_k)
    V = apply_linear_projection(x, w_v, b_v)
    return Q, K, V

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))
    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    merged_heads = merge_heads_back_to_model_dim(context)
    projections = apply_linear_projection(merged_heads, w_o, b_o)
    return projections

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)
    scaled_multi_head_attention, _ = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask)
    merged_scaled_multi_head_attention = merge_heads_and_project_output(scaled_multi_head_attention, w_o, None)
    return merged_scaled_multi_head_attention

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    linear_projection = x @ w1 + b1
    torch.nn.functional.relu(linear_projection, inplace=True)
    return linear_projection

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    first_layer_activations = apply_ffn_first_linear_and_relu(x, w1, b1)
    second_layer_activations = apply_ffn_second_linear(first_layer_activations, w2, b2)
    return second_layer_activations

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, correction=0)
    return mean, var

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean, var = compute_layer_norm_mean_and_variance(x)
    x_normalized = (x - mean) / (torch.sqrt(var + eps))
    y = gamma * x_normalized + beta
    return y

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    add = residual_input + sublayer_output
    normalized_residual_addition = normalize_and_scale_with_gamma_beta(add, gamma, beta, eps)
    return normalized_residual_addition

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    x_masked = x.masked_fill(~keep_mask, value=0.0)
    x_masked /= keep_prob
    return x_masked

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    merged_scaled_multi_head_attention = assemble_multi_head_attention_forward(
        x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask,
    )
    normalized_residual = apply_residual_add_and_norm(x, merged_scaled_multi_head_attention, gamma, beta)
    return normalized_residual

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    position_wise_ffn = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    normalized_position_wise_ffn = apply_residual_add_and_norm(x, position_wise_ffn, gamma, beta)
    return normalized_position_wise_ffn

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    p = layer_params
    w_q, w_k, w_v, w_o = (
        p["w_q"], p["w_k"], 
        p["w_v"], p["w_o"],
    )
    mha = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    attn_gamma, attn_beta = p["attn_gamma"], p["attn_beta"]
    h = apply_residual_add_and_norm(x, mha, attn_gamma, attn_beta)
    w1, b1, w2, b2, ffn_gamma, ffn_beta = (
        p["w1"], p["b1"],
        p["w2"], p["b2"],
        p["ffn_gamma"], p["ffn_beta"],
    )
    y = encoder_layer_feed_forward_sublayer(h, w1, b1, w2, b2, ffn_gamma, ffn_beta)
    return y

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    h = x.clone()
    for layer in encoder_layer_params_list:
        h = assemble_encoder_layer(h, layer, num_heads, src_mask)
    return h

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    mha = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    result = apply_residual_add_and_norm(y, mha, gamma, beta)
    return result

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask[:, None, None, :]
    cross_attention = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    normalized_scores = apply_residual_add_and_norm(y, cross_attention, gamma, beta)
    return normalized_scores

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    ffn = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    residuals = apply_residual_add_and_norm(y, ffn, gamma, beta)
    return residuals

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    p = layer_params

    h = decoder_layer_masked_self_attention_sublayer(
        y,
        p["w_q_self"], p["w_k_self"], p["w_v_self"], p["w_o_self"],
        p["self_gamma"], p["self_beta"],
        num_heads, tgt_mask,
    )

    h = decoder_layer_cross_attention_sublayer(
        h, encoder_output,
        p["w_q_cross"], p["w_k_cross"], p["w_v_cross"], p["w_o_cross"],
        p["cross_gamma"], p["cross_beta"],
        num_heads, src_mask,
    )

    return decoder_layer_feed_forward_sublayer(
        h,
        p["w1"], p["b1"], p["w2"], p["b2"],
        p["ffn_gamma"], p["ffn_beta"],
    )

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    h = y.clone()
    for params in decoder_layer_params_list:
        h = assemble_decoder_layer(h, encoder_output, params, num_heads, src_mask, tgt_mask)
    return h

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.T

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return torch.nn.functional.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    if "src_embedding" in model_params:
        token_embedding_src = model_params["src_embedding"]
        token_embedding_tgt = model_params["tgt_embedding"]
    else:
        token_embedding_src = model_params["token_embedding"]
        token_embedding_tgt = model_params["token_embedding"]
    
    output_projection = model_params["output_projection"]
    d_model = token_embedding_src.shape[1]

    src_len, tgt_len = src_ids.shape[1], tgt_ids.shape[1]
    pe = build_sinusoidal_positional_encoding(max(src_len, tgt_len), d_model)

    src_emb = scale_embeddings_by_sqrt_d_model(token_embedding_src[src_ids], d_model)
    src_emb = add_positional_encoding_to_embeddings(src_emb, pe)

    tgt_emb = scale_embeddings_by_sqrt_d_model(token_embedding_tgt[tgt_ids], d_model)
    tgt_emb = add_positional_encoding_to_embeddings(tgt_emb, pe)

    src_mask = build_padding_mask(src_ids, pad_id)
    tgt_pad_mask = build_padding_mask(tgt_ids, pad_id)
    causal = build_causal_mask(tgt_len)
    tgt_mask = combine_padding_and_causal_masks(tgt_pad_mask, causal)

    encoder_output = stack_encoder_layers(
        src_emb, model_params["encoder_layers"], num_heads, src_mask,
    )
    decoder_output = stack_decoder_layers(
        tgt_emb, encoder_output, model_params["decoder_layers"],
        num_heads, src_mask, tgt_mask,
    )

    logits = apply_final_output_projection(decoder_output, output_projection)
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    w_q = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_k = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_v = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_o = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)

    w1 = torch.rand(d_model, d_ff, dtype=torch.float32, requires_grad=True)
    b1 = torch.zeros(d_ff, dtype=torch.float32, requires_grad=True)
    w2 = torch.rand(d_ff, d_model, dtype=torch.float32, requires_grad=True)
    b2 = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    attn_gamma = torch.ones((d_model, ), dtype=torch.float32, requires_grad=True)
    attn_beta = torch.zeros((d_model, ), dtype=torch.float32, requires_grad=True)

    ffn_gamma = torch.ones((d_model, ), dtype=torch.float32, requires_grad=True)
    ffn_beta = torch.zeros((d_model, ), dtype=torch.float32, requires_grad=True)

    return dict(
        w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o,
        w1=w1, w2=w2, b1=b1, b2=b2,
        attn_gamma=attn_gamma, attn_beta=attn_beta,
        ffn_gamma=ffn_gamma, ffn_beta=ffn_beta,
    )

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    w_q_self = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_k_self = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_v_self = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_o_self = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)

    w_q_cross = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_k_cross = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_v_cross = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)
    w_o_cross = torch.rand(d_model, d_model, dtype=torch.float32, requires_grad=True)

    w1 = torch.rand(d_model, d_ff, dtype=torch.float32, requires_grad=True)
    b1 = torch.zeros(d_ff, dtype=torch.float32, requires_grad=True)
    w2 = torch.rand(d_ff, d_model, dtype=torch.float32, requires_grad=True)
    b2 = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    self_gamma = torch.ones((d_model, ), dtype=torch.float32, requires_grad=True)
    self_beta = torch.zeros((d_model, ), dtype=torch.float32, requires_grad=True)

    cross_gamma = torch.ones((d_model, ), dtype=torch.float32, requires_grad=True)
    cross_beta = torch.zeros((d_model, ), dtype=torch.float32, requires_grad=True)

    ffn_gamma = torch.ones((d_model, ), dtype=torch.float32, requires_grad=True)
    ffn_beta = torch.zeros((d_model, ), dtype=torch.float32, requires_grad=True)

    return dict(
        w_q_self=w_q_self, w_k_self=w_k_self, w_v_self=w_v_self, w_o_self=w_o_self,
        w_q_cross=w_q_cross, w_k_cross=w_k_cross, w_v_cross=w_v_cross, w_o_cross=w_o_cross,
        w1=w1, w2=w2, b1=b1, b2=b2,
        self_gamma=self_gamma, self_beta=self_beta,
        cross_gamma=cross_gamma, cross_beta=cross_beta,
        ffn_gamma=ffn_gamma, ffn_beta=ffn_beta,
    )

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    src_embedding = torch.rand(vocab_size, d_model, dtype=torch.float32, requires_grad=True)
    tgt_embedding = torch.rand(vocab_size, d_model, dtype=torch.float32, requires_grad=True)
    output_projection = (
        tgt_embedding if tie_weights
        else torch.rand(
            vocab_size, d_model, 
            dtype=torch.float32, requires_grad=True
        )
    )
    return dict(
        src_embedding=src_embedding,
        tgt_embedding=tgt_embedding,
        output_projection=output_projection,
    )

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    layers = []
    ids = set()

    params = (*encoder_layer_params, *decoder_layer_params, embedding_params)
    for name_param in params:
        for layer in name_param.values():
            if id(layer) not in ids:
                ids.add(id(layer))
                layers.append(layer)
    return layers

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    start_token_tensor = torch.full_like(target_ids[..., :1], start_token_id)
    shifted_ids = torch.cat(
        (start_token_tensor, target_ids[..., :-1]), dim=-1,
    )
    return shifted_ids

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    return d_model ** (-1 / 2) * min(step ** (-1 / 2), step * warmup_steps ** (-3 / 2))

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    return torch.full(shape, fill_value=epsilon / (vocab_size - 2))

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    gold_token_ids_tensor = gold_token_ids.clone().unsqueeze(-1)
    return torch.scatter(smoothed_distribution, index=gold_token_ids_tensor, dim=-1, value=confidence)

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    pad_tensor = smoothed_distribution.clone()

    pad_tensor[..., pad_id] = 0.0

    mask = (gold_token_ids == pad_id).unsqueeze(-1)
    pad_tensor.masked_fill_(mask, 0.0)
    return pad_tensor

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    return torch.sum(-torch.sum(log_probabilities * smoothed_distribution, dim=-1))

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    not_pad_ids_count = (gold_token_ids != pad_id).sum()
    return total_loss / max(not_pad_ids_count.item(), 1)

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    not_pad_ids = (gold_token_ids != pad_id).float()
    number_of_not_pad_ids = not_pad_ids.sum()
    if number_of_not_pad_ids == 0:
        return number_of_not_pad_ids
    predicted_tokens = torch.argmax(log_probabilities, dim=-1)
    hitted_gold_tokens = (predicted_tokens == gold_token_ids).float()
    accuracy = (hitted_gold_tokens * not_pad_ids).sum() / number_of_not_pad_ids
    return accuracy

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    state = {}
    state["t"] = 0
    state["m"] = []
    state["v"] = []
    for key in ("m", "v"):
        for param in parameter_list:
            state[key].append(torch.zeros_like(param))
    return state

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    return beta1 * m_prev + (1 - beta1) * grad.detach()

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    return beta2 * v_prev + (1 - beta2) * grad.detach() ** 2

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    m_hat = m_t / (1 - beta1 ** step)
    v_hat = v_t / (1 - beta2 ** step)
    return m_hat, v_hat

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    optimizer_state["t"] += 1
    step = optimizer_state["t"]
    with torch.no_grad():
        for idx, param in enumerate(parameter_list):
            grad = param.grad
            if grad is None:
                continue
            m_prev = optimizer_state["m"][idx]
            v_prev = optimizer_state["v"][idx]
            m_t = update_adam_first_moment(m_prev, grad, beta1)
            v_t = update_adam_second_moment(v_prev, grad, beta2)
            m_hat, v_hat = apply_adam_bias_correction(m_t, v_t, beta1, beta2, step)
            param.data -= learning_rate * m_hat / (v_hat ** 0.5 + epsilon)
            optimizer_state["m"][idx] = m_t
            optimizer_state["v"][idx] = v_t

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for param in parameter_list:
        param.grad = None

# Step 71 - compute_batch_training_loss
"""
model_params:
    encoder_layers 
    decoder_layers 
    src_embedding 
    tgt_embedding 
    output_projection
config:
    pad_id
    start_id
    vocab_size
    smoothing
    d_model 
    num_heads
"""

def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    # TODO: shift targets right, run the forward pass, build smoothed targets, and average the KL loss over non-pad tokens.
    tgt_ids = shift_targets_right_with_start_token(tgt_batch, config["start_id"])
    logits = run_transformer_forward(src_batch, tgt_ids, model_params, config["num_heads"], config["pad_id"])
    
    confidence = 1 - config["smoothing"]
    smoothing_distr = build_uniform_smoothing_distribution((*tgt_batch.shape, config["vocab_size"]), config["vocab_size"], config["smoothing"])
    confidence_smoothing_distr = set_confidence_on_gold_tokens(smoothing_distr, tgt_ids, confidence)
    zero_pad_distr = zero_pad_column_and_pad_token_rows(confidence_smoothing_distr,  tgt_ids, config["pad_id"])

    smoothed_kl_loss = compute_label_smoothed_kl_loss(logits, zero_pad_distr)
    loss = average_loss_over_non_pad_tokens(smoothed_kl_loss,  tgt_ids, config["pad_id"])
    model_params.setdefault("token_embedding", model_params.get("src_embedding"))
    return loss

# Step 72 - run_training_step_with_backprop
import torch

"""
model_params:
    src_embedding tgt_embedding output_projection encoder_layers decoder_layers

optimizer_state:
    t m v

config:
    pad_id start_id vocab_size smoothing num_heads d_model warmup_steps
"""


def run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """
    # TODO: zero grads, compute loss, backward, look up Noam LR, apply Adam step
    zero_all_parameter_gradients(parameter_list)

    loss = compute_batch_training_loss(src_batch, tgt_batch, model_params, config)
    loss.backward()

    noam_lr = compute_noam_learning_rate(step_number, config["d_model"], config["warmup_steps"])
    optimizer_state = apply_adam_step_to_all_parameters(parameter_list, optimizer_state, noam_lr)

    return loss.item()

# Step 73 - run_training_loop_for_steps
def run_training_loop_for_steps(batches, parameter_list, model_params, optimizer_state, num_steps, config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    # TODO: iterate for num_steps steps, calling run_training_step_with_backprop each time
    loss_lst = []
    for t in range(1, num_steps + 1):
        batch = batches[(t - 1) % len(batches)]
        loss = run_training_step_with_backprop(*batch, parameter_list, model_params, optimizer_state, t, config)
        loss_lst.append(loss)
    return loss_lst

# Step 74 - pick_next_token_by_argmax
import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.

    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    # TODO: pick the next greedy token id by taking the argmax over the vocab axis
    return torch.argmax(final_step_logits, dim=-1)

# Step 75 - compute_length_penalty
def compute_length_penalty(sequence_length, alpha):
    # TODO: return the Google NMT length penalty for the given sequence_length and alpha.
    return ((5 + sequence_length) / 6) ** alpha

# Step 76 - compute_candidate_scores
import torch

def compute_candidate_scores(beam_scores, next_token_log_probs):
    # TODO: add each beam's running log-prob to its row of next-token log probs.
    beam_scores = beam_scores.unsqueeze(-1)
    return next_token_log_probs + beam_scores

# Step 77 - select_top_k_candidates
import torch

def select_top_k_candidates(candidate_scores, k):
    # TODO: pick the top k (beam_index, token_id, score) triples from candidate_scores
    vocab_size = candidate_scores.shape[-1]

    candidate_scores = candidate_scores.flatten()
    top_k = torch.topk(candidate_scores, k)

    scores = top_k.values
    indices = top_k.indices

    beam_indices = indices // vocab_size
    token_ids = indices % vocab_size

    return {
        "beam_indices": beam_indices,
        "token_ids": token_ids,
        "scores": scores,
    }

# Step 78 - append_tokens_to_beam_sequences
import torch

def append_tokens_to_beam_sequences(beam_sequences, beam_indices, token_ids):
    # TODO: gather parent beam rows and append the new token ids as the last column
    parent_rows = beam_sequences[beam_indices, :]
    beam_rows = torch.cat((parent_rows, token_ids.unsqueeze(-1)), dim=-1)
    return beam_rows

# Step 79 - mark_finished_beams
import torch

def mark_finished_beams(token_ids, finished_flags, end_token_id):
    # TODO: return updated boolean finished flags for each beam given the new token ids
    return (token_ids == end_token_id).masked_fill(finished_flags, True)

# Step 80 - select_best_finished_beam
def select_best_finished_beam(finished_sequences, finished_scores, alpha):
    # TODO: return the finished beam with the highest length-penalized score
    max_google_score = -float("inf")
    max_index = -1
    for idx, seq in enumerate(finished_sequences):
        y_i = finished_scores[idx] / compute_length_penalty(len(seq), alpha)
        if y_i > max_google_score:
            max_google_score = y_i
            max_index = idx
    return dict(
        sequence=finished_sequences[max_index],
        score=max_google_score,
    )

