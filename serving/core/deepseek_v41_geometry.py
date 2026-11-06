"""Logical tensor boundaries for the existing DeepSeek V4.1 operator catalog."""

BOUNDARIES = frozenset(('engram_hash', 'engram_lookup', 'input_projections',
    'qkv_norm', 'q_b_proj', 'swa_rope_cache', 'compress_state_norm',
    'compressed_cache_write', 'indexer_prepare', 'indexer_attention'))


def logical_sizes(config, name, rows, tp, fp, layer_index):
    """Do not infer packed storage or sparse cache traffic from logical shapes.

    Cache occupancy comes from the native arena. Parameters of quantized
    operators may be replaced with observed weight and scale bytes at emission.
    """
    if type(rows) is not int or rows < 0 or type(tp) is not int or tp < 1:
        raise ValueError('Invalid V4.1 token rows or TP size')
    if name in ('engram_hash', 'engram_lookup'):
        # NgramHashState emits int32 [tokens, layers, hash_columns]. The
        # NVIDIA table lookup stages BF16 rows; CPU storage is counted apart.
        heads = (config['engram_max_ngram_size']-1)*config['engram_n_heads']
        n = len(config['engram_layer_ids']) * rows * heads
        if name == 'engram_hash':
            return 16*rows, 0, 4*n  # int64 token IDs and positions
        return 4*n, 0, n*config['engram_head_dim']*fp

    if (type(layer_index) is not int
            or not 0 <= layer_index < config['num_hidden_layers']):
        raise ValueError('V4.1 attention sizing requires the decoder layer index')
    h, d, q, heads = (config[key] for key in
                       ('hidden_size', 'head_dim', 'q_lora_rank', 'num_attention_heads'))
    if heads % tp:
        raise ValueError('V4.1 attention heads must divide TP')
    local_q = heads//tp*d
    kv_source = layer_index in config['kv_source_layer_ids']
    index_source = layer_index in config['index_source_layer_ids']
    ratio = config['compress_ratios'][layer_index]
    ih, idim = config['index_n_heads'], config['index_head_dim']
    score_cols = d*(1+int(ratio > 1)) if kv_source else 0

    if name == 'input_projections':
        # Replicated fused Q/KV, compressor score and indexer weight GEMMs.
        weights = h*(q+d+score_cols+(ih if index_source else 0))*fp
        out = rows*((q+d+(ih if index_source else 0))*fp + score_cols*4)
        return rows*h*fp, weights, out
    if name == 'qkv_norm':
        return rows*(q+d)*fp, (q+d)*fp, rows*(q+d)*fp
    if name == 'q_b_proj':
        return rows*q*fp, q*local_q*fp, rows*local_q*fp
    if name == 'swa_rope_cache':
        return rows*((local_q+d)*fp+8), 0, rows*local_q*fp
    if name in ('compress_state_norm', 'compressed_cache_write'):
        if not kv_source or ratio not in (1, 2):
            raise ValueError('V4.1 compressor runs only on a compressed KV source')
        if name == 'compress_state_norm':
            return rows*(score_cols*4+8), d*fp, rows*d*fp
        # insert_cache writes native state and returns no activation tensor.
        return rows*(d*fp+8), 0, 0
    if name == 'indexer_prepare':
        if not index_source:
            raise ValueError('V4.1 indexer preparation requires an index source')
        input_width = q + ih + (d if kv_source else 0)
        weights = q*ih*idim + ((d+1)*idim if kv_source else 0)
        # Logical upper bound: the short-context branch can skip query work.
        return rows*(input_width*fp+8), weights*fp, rows*(ih*idim*fp+ih*4)
    if name == 'indexer_attention':
        index_query = (ih*idim*fp+ih*4) if index_source else 0
        return rows*(local_q*fp+index_query), (heads//tp)*4, rows*local_q*fp
    raise ValueError(f'Unknown V4.1 logical boundary: {name}')
