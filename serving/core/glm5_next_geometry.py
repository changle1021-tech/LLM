"""Source-backed logical tensor and cache geometry for GLM-5.3-Flash.

The boundaries follow vLLM 0.31 ``models/glm5next/common/{model,kda,attention}``
and ``model_executor/layers/{mla,mhc}``. Logical referenced tensors exclude
allocator padding, workspaces, quantization scales and cached parameter copies;
they do not establish resident capacity or per-kernel memory traffic.
"""


MODEL_TYPES = frozenset(('glm5_next', 'glm5_next_text'))
BOUNDARIES = frozenset((
    'embedding', 'final_layernorm', 'lm_head', 'sampler',
    'mhc_expand', 'mhc_pre', 'mhc_fused_post_pre', 'mhc_post', 'mhc_contract',
    'kda_in_proj', 'kda_f_b_proj', 'kda_g_b_proj', 'kda_core',
    'kda_o_norm', 'kda_o_proj', 'mla_in_proj', 'mla_qkv_norm',
    'mla_q_b_proj', 'mla_o_proj', 'indexer_wq_b_proj',
    'indexer_wk_weights_proj', 'indexer_glue', 'indexer', 'attention',
    'gate_up_proj', 'act_fn', 'down_proj', 'moe',
))


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError('GLM-5.3 ' + name + ' must be a positive integer')
    return value


def _nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError('GLM-5.3 ' + name + ' must be a nonnegative integer')
    return value


def text_config(config):
    """Read the native backbone while retaining wrapper quantization policy."""
    if config.get('model_type') not in MODEL_TYPES:
        raise ValueError('GLM-5.3 geometry requires the native GLM5Next config')
    inner = config.get('text_config', config)
    if not isinstance(inner, dict):
        raise ValueError('GLM-5.3 text_config must be a mapping')
    result = dict(inner)
    for name in ('quantization_config', 'head_dtype', 'mamba_cache_dtype',
                 'mamba_ssm_dtype'):
        if name in config:
            result[name] = config[name]
    return result


def _geometry(config, tp):
    cfg = text_config(config)
    _positive(tp, 'TP')
    if cfg.get('dtype', cfg.get('torch_dtype')) not in ('bfloat16', 'bf16'):
        raise NotImplementedError('GLM-5.3 logical geometry requires BF16 activations')
    if cfg.get('head_dtype') not in (None, 'bfloat16', 'bf16'):
        raise NotImplementedError('GLM-5.3 alternate head dtype requires its own contract')
    h = _positive(cfg.get('hidden_size'), 'hidden_size')
    linear = cfg.get('linear_attn_config') or {}
    if not isinstance(linear, dict):
        raise ValueError('GLM-5.3 linear_attn_config must be a mapping')
    heads = _positive(linear.get('num_heads', cfg.get('linear_num_heads')), 'KDA heads')
    d = _positive(linear.get('head_dim', cfg.get('linear_head_dim')), 'KDA head width')
    conv = _positive(linear.get('short_conv_kernel_size',
                                cfg.get('linear_conv_kernel_dim')), 'KDA convolution width')
    ah = _positive(cfg.get('num_attention_heads'), 'MLA heads')
    if heads % tp or ah % tp:
        raise ValueError('GLM-5.3 KDA and MLA head counts must divide TP')
    if cfg.get('mla_use_nope', cfg.get('mla_nope', True)) is not True or cfg.get('qk_rope_head_dim') != 0:
        raise NotImplementedError('GLM-5.3 geometry covers the NoPE MLA checkpoint')
    if cfg.get('mhc') is not True:
        raise NotImplementedError('GLM-5.3 geometry requires its mHC execution path')
    return cfg, h, heads // tp, d, conv, ah // tp


def logical_sizes(config, name, rows, tp=1, layer_index=None, *, ep=1):
    """Return per-rank logical ``(input, parameter, output)`` byte counts.

    Activations, KDA and MLA matrices are BF16 even in the FP8 checkpoint.
    mHC mixing parameters and KDA convolution/gate coefficients are FP32.
    FP8 MLP bytes count matrix payloads, independently of runtime scale/padding
    storage. ``ep`` shards routed experts only; shared MLP matrices shard by TP.
    With EP disabled, routed experts shard their intermediate width by TP.
    ``layer_index`` is accepted to match the common trace-sizing interface.
    Cache contents and position metadata are kept out of component triples.
    """
    _nonnegative(rows, 'token rows')
    cfg, h, kh, d, conv, ah = _geometry(config, tp)
    _positive(ep, 'EP')
    if name not in BOUNDARIES:
        raise ValueError('Unknown GLM-5.3 logical boundary: ' + str(name))
    p = kh * d
    if name == 'kda_in_proj':
        width = 3 * p + kh + 2 * d
        return rows * h * 2, h * width * 2, rows * width * 2
    if name in ('kda_f_b_proj', 'kda_g_b_proj'):
        return rows * d * 2, d * p * 2, rows * p * 2
    if name == 'kda_core':
        # _forward consumes q|k|v, f_b output and raw beta. The persistent
        # state is accounted per sequence, not as a token-dependent read.
        weights = (3 * p * conv + p + kh) * 4
        return rows * (4 * p + kh) * 2, weights, rows * p * 2
    if name == 'kda_o_norm':
        return rows * p * 4, d * 2, rows * p * 2
    if name == 'kda_o_proj':
        return rows * p * 2, p * h * 2, rows * h * 2

    hc = _positive(cfg.get('hc_mult', cfg.get('mhc_num_residual_streams')), 'mHC streams')
    residual = hc * h
    mix = hc * (hc + 2)
    coefficients = hc + hc * hc
    if name == 'mhc_expand':
        return rows * h * 2, 0, rows * residual * 2
    if name == 'mhc_contract':
        return rows * residual * 2, 0, rows * h * 2
    if name == 'mhc_pre':
        weights = (mix * residual + mix + 3) * 4 + h * 2
        return rows * residual * 2, weights, rows * (h * 2 + coefficients * 4)
    if name == 'mhc_fused_post_pre':
        io = rows * ((h + residual) * 2 + coefficients * 4)
        weights = (mix * residual + mix + 3) * 4 + h * 2
        return io, weights, io
    if name == 'mhc_post':
        return rows * ((h + residual) * 2 + coefficients * 4), 0, rows * residual * 2

    q = _positive(cfg.get('q_lora_rank'), 'query latent width')
    latent = _positive(cfg.get('kv_lora_rank'), 'KV latent width')
    nope = _positive(cfg.get('qk_nope_head_dim'), 'NoPE head width')
    value = _positive(cfg.get('v_head_dim'), 'value head width')
    if name == 'mla_in_proj':
        return rows * h * 2, h * (q + latent) * 2, rows * (q + latent) * 2
    if name == 'mla_qkv_norm':
        return rows * (q + latent) * 2, (q + latent) * 2, rows * (q + latent) * 2
    if name == 'mla_q_b_proj':
        return rows * q * 2, q * ah * nope * 2, rows * ah * nope * 2
    if name == 'mla_o_proj':
        return rows * ah * value * 2, ah * value * h * 2, rows * h * 2
    if name == 'attention':
        # kv_b_proj is consumed by the absorbed native MLA implementation.
        # It is resident once, not another executed linear catalog entry.
        weights = latent * ah * (nope + value) * 2
        return rows * (ah * nope + latent) * 2, weights, rows * ah * value * 2

    ih = _positive(cfg.get('index_n_heads'), 'indexer heads')
    index_dim = _positive(cfg.get('index_head_dim'), 'indexer head width')
    pool = _positive(cfg.get('index_kpool'), 'indexer pool width')
    if index_dim != 128 or pool <= 1 or any(cfg.get(key, True) is not True for key in
            ('index_kpool_compress', 'index_kpool_always_select_tail', 'index_dsa_use_layernorm')):
        raise NotImplementedError('GLM-5.3 geometry requires its native 128-wide pooled indexer')
    padded_ih = max(ih, 32)
    if padded_ih not in (32, 64):
        raise NotImplementedError('GLM-5.3 indexer heads require the native 32/64 padded layout')
    if name == 'indexer_wq_b_proj':
        return rows * q * 2, q * ih * index_dim * 2, rows * ih * index_dim * 2
    if name == 'indexer_wk_weights_proj':
        width = index_dim + ih
        return rows * h * 2, h * width * 2, rows * width * 2
    if name == 'indexer_glue':
        # Includes raw compress-gate projection and FP32 LayerNorm weights.
        # The cached FP32 head-gate copy derives from checkpoint parameters
        # owned by wk_weights_proj; it is additional buffer residency, not
        # another parameter tensor in this logical boundary.
        weights = h * index_dim * 2 + index_dim * 8
        inp = rows * (h + ih * index_dim + index_dim) * 2
        out = rows * (padded_ih * index_dim + 4 * index_dim + padded_ih * 4)
        return inp, weights, out
    if name == 'indexer':
        topk = _positive(cfg.get('index_topk'), 'indexer top-k')
        if topk % pool:
            raise ValueError('GLM-5.3 top-k must divide the pool width')
        width = ((topk + pool - 1 + 127) // 128) * 128
        # q is FP8, K and gate are BF16, head weights are FP32. Input hidden
        # states supply shape/device only and are not read by the native op.
        inp = rows * (padded_ih * index_dim + 4 * index_dim + padded_ih * 4)
        return inp, pool * index_dim * 4, rows * width * 4

    vocab = _positive(cfg.get('vocab_size'), 'vocabulary')
    padded_vocab = ((vocab + 63) // 64) * 64
    if padded_vocab % tp:
        raise ValueError('GLM-5.3 padded vocabulary must divide TP')
    local_vocab = padded_vocab // tp
    if name == 'embedding':
        # The v0.31 V1 runner owns int32 input_ids and int64 positions.
        return rows * 4, local_vocab * h * 2, rows * h * 2
    if name == 'final_layernorm':
        return rows * h * 2, h * 2, rows * h * 2
    if name == 'lm_head':
        weights = 0 if cfg.get('tie_word_embeddings') else local_vocab * h * 2
        return rows * h * 2, weights, rows * local_vocab * 2
    if name == 'sampler':
        # V1 Sampler consumes BF16 logits and casts internally to FP32;
        # sampler.py:138-146 returns one int32 token ID per sequence.
        return rows * vocab * 2, 0, rows * 4

    quant = cfg.get('quantization_config') or {}
    if quant.get('quant_method') not in (None, 'fp8'):
        raise NotImplementedError('GLM-5.3 MLP geometry covers BF16 or native FP8 payloads')
    matrix = 1 if quant.get('quant_method') == 'fp8' else 2
    ffn = _positive(cfg.get('intermediate_size'), 'dense MLP width')
    if ffn % tp:
        raise ValueError('GLM-5.3 dense MLP width must divide TP')
    if name == 'gate_up_proj':
        return rows * h * 2, 2 * h * (ffn // tp) * matrix, rows * (ffn // tp) * 4
    if name == 'act_fn':
        return rows * (ffn // tp) * 4, 0, rows * (ffn // tp) * 2
    if name == 'down_proj':
        return rows * (ffn // tp) * 2, (ffn // tp) * h * matrix, rows * h * 2
    if name == 'moe':
        experts = _positive(cfg.get('n_routed_experts'), 'routed experts')
        shared = _nonnegative(cfg.get('n_shared_experts', 0), 'shared experts')
        width = _positive(cfg.get('moe_intermediate_size'), 'expert width')
        if experts % ep or width % tp:
            raise ValueError('GLM-5.3 routed experts must divide EP and expert width must divide TP')
        # This describes non-SP logical matrices. Native component tables own
        # dispatch topology, runtime packing and shared/routed kernel timing.
        routed_width = width if ep > 1 else width // tp
        weights = (h * experts * 2 + experts * 4
                   + (experts // ep) * 3 * h * routed_width * matrix
                   + shared * 3 * h * (width // tp) * matrix)
        return rows * h * 2, weights, rows * h * 2
    raise ValueError('Unknown GLM-5.3 logical boundary: ' + str(name))


def cache_bytes(config, tokens, tp=1, *, linear=False, num_spec_tokens=0):
    """Return ``(token_cache_bytes, fixed_sequence_bytes)`` for one layer.

    MLA caches are replicated: BF16 latent rows, capacity for one FP8+FP32-scale
    index record per pool (rounding a final partial pool up), and a BF16 tail
    ring. The incomplete pool's contents reside in the tail, not the index.
    KDA stores TP-sharded convolution and FP32 recurrent state per sequence.
    These are logical payloads, before the native allocator's page alignment.
    """
    _nonnegative(tokens, 'cached tokens')
    _nonnegative(num_spec_tokens, 'speculative tokens')
    if type(linear) is not bool:
        raise ValueError('GLM-5.3 linear-cache selection must be a boolean')
    cfg, _, heads, d, conv, _ = _geometry(config, tp)
    if linear:
        if cfg.get('mamba_cache_dtype', 'auto') not in ('auto', 'bfloat16', 'bf16'):
            raise NotImplementedError('GLM-5.3 KDA convolution geometry requires BF16 state')
        if cfg.get('mamba_ssm_dtype', 'auto') not in ('auto', 'float32', 'fp32'):
            raise NotImplementedError('GLM-5.3 KDA geometry requires native FP32 recurrent state')
        fixed = 3 * heads * d * (conv - 1 + num_spec_tokens) * 2 + heads * d * d * 4
        return 0, fixed
    quant = cfg.get('quantization_config') or {}
    if (cfg.get('kv_cache_dtype', 'auto') not in ('auto', 'bfloat16', 'bf16')
            or quant.get('kv_cache_scheme') is not None
            or quant.get('kv_cache_quant_algo') is not None):
        raise NotImplementedError('GLM-5.3 cache geometry requires the BF16 MLA cache layout')
    latent = _positive(cfg.get('kv_lora_rank'), 'KV latent width')
    index_dim = _positive(cfg.get('index_head_dim'), 'indexer head width')
    pool = _positive(cfg.get('index_kpool'), 'indexer pool width')
    if index_dim % 128 or pool <= 1:
        raise NotImplementedError('GLM-5.3 pooled cache needs 128-element index scale blocks')
    index_record = index_dim + (index_dim // 128) * 4
    token_bytes = tokens * latent * 2 + ((tokens + pool - 1) // pool) * index_record
    ring_pools = (pool + num_spec_tokens + pool - 1) // pool
    ring = pool * (1 << (ring_pools - 1).bit_length())
    return token_bytes, ring * 2 * index_dim * 2
