"""Tensor/collective contracts, independent of latency measurements."""
import math


def dtype_bytes(dtype, default=2):
    if dtype is None or str(dtype) == "auto":
        return default
    name = str(dtype)
    if name.startswith("torch."):
        name = name[6:]
    sizes = {"bfloat16": 2, "bf16": 2, "float16": 2, "half": 2,
             "float32": 4, "float": 4, "fp32": 4, "int32": 4, "int64": 8,
             "float8_e4m3fn": 1, "float8_e5m2": 1, "uint8": 1}
    if name not in sizes:
        raise ValueError(f"Unsupported communication dtype: {dtype}")
    return sizes[name]


def moe_dispatch_payload_bytes(contract):
    """Per-row tensor payloads at the native pre-AllGather boundary.

    Without a recorded layout, components dispatch hidden states and two
    top-k tensors. A recorded layout lists every row tensor with its dtype.
    """
    h, k = contract['hidden_dim'], contract['global_top_k']
    if any(type(value) is not int or value < 1 for value in (h, k)):
        raise ValueError('MoE dispatch requires positive hidden and top-k dimensions')
    layout = contract.get('dispatch_layout')
    if layout is None:
        return (h*dtype_bytes(contract['input_dtype']),
                k*dtype_bytes(contract['topk_weights_dtype']),
                k*dtype_bytes(contract['topk_ids_dtype']))
    tensors = layout.get('tensors') if isinstance(layout, dict) else None
    if not isinstance(tensors, list) or not tensors:
        raise ValueError('Native MoE dispatch layout lists no row tensors')
    payloads = []
    for tensor in tensors:
        shape = tensor.get('row_shape') if isinstance(tensor, dict) else None
        if (not isinstance(shape, list)
                or any(type(n) is not int or n < 1 for n in shape)):
            raise ValueError('Native MoE dispatch tensor needs a positive row shape')
        payloads.append(math.prod(shape)*dtype_bytes(tensor['dtype']))
    return tuple(payloads)


def vocab_shard_size(config, tp, padding=64):
    """vLLM pads vocabulary before partitioning it, not after."""
    padded = ((int(config["vocab_size"]) + padding - 1) // padding) * padding
    if tp < 1 or padded % tp:
        raise ValueError(f"Padded vocabulary {padded} is not divisible by TP={tp}")
    return padded // tp


def sequence_parallel_rows(rows, tp):
    """Return local and padded token rows for native dim-0 SP collectives."""
    if type(rows) is not int or rows < 1 or type(tp) is not int or tp < 1:
        raise ValueError('Sequence parallelism requires positive token rows and TP')
    local = (rows + tp - 1) // tp
    return local, local * tp


def engram_tp_geometry(config, tp, *, sequence_parallel=False):
    """Native V4.1 Engram geometry without EDP sharding.

    common/engram.py pads head shards before gathering, then slices to the
    original head count. The column-parallel projection divides its output
    width exactly. These are logical dimensions, not packed weight storage.
    """
    if config.get('model_type') not in ('deepseek_v41', 'deepseek_v41_text'):
        raise ValueError('Engram TP geometry requires a DeepSeek V4.1 text config')
    fields = ('engram_max_ngram_size', 'engram_n_heads', 'engram_head_dim',
              'hidden_size', 'hc_mult')
    if (type(tp) is not int or tp < 1
            or any(type(config.get(key)) is not int or config[key] < 1 for key in fields)
            or config['engram_max_ngram_size'] < 2):
        raise ValueError('Invalid Engram dimensions or TP degree')
    heads = (config['engram_max_ngram_size'] - 1) * config['engram_n_heads']
    local_heads = (heads + tp - 1) // tp
    width = (config['hc_mult'] + 1) * config['hidden_size']
    if (tp - 1) * local_heads >= heads or (not sequence_parallel and width % tp):
        raise ValueError('Engram TP leaves an empty head shard or does not divide wkv output')
    return dict(heads=heads, local_heads=local_heads,
                embed_width=heads * config['engram_head_dim'],
                output_width=width,
                local_output_width=width if sequence_parallel else width // tp)
