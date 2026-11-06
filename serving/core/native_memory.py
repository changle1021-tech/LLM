"""Admission for an observed shared native cache arena on a bounded workload."""
from dataclasses import dataclass
import hashlib
import json


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class NativePagePool:
    resident_gpu_bytes: int
    native_pages: int
    page_stride_bytes: int
    block_size: int
    pages_per_logical_block: int
    fixed_pages: int
    max_model_len: int
    max_num_seqs: int
    kv_cache_dtype: str

    @property
    def logical_block_bytes(self):
        return self.page_stride_bytes * self.pages_per_logical_block

    @property
    def logical_blocks(self):
        return (self.native_pages - self.fixed_pages) // self.pages_per_logical_block

    @property
    def arena_bytes(self):
        return self.native_pages * self.page_stride_bytes

    def kv_bytes(self, tokens):
        if type(tokens) is not int or tokens < 0:
            raise ValueError('Native KV token count must be a nonnegative integer')
        return (tokens*self.logical_block_bytes + self.block_size - 1)//self.block_size


def native_page_pool(record, source_config, runtime):
    """Use measured residency and reserve every group's full-history pages.

    This flattens capacity conservatively, without claiming to reproduce SWA
    reclaim or bounded prefix replay. Admission guarantees enough pages even
    if all active sequences reach the declared context limit concurrently.
    """
    if (not isinstance(record, dict) or record.get('schema') != 'native-v41-shared-page-pool'
            or record.get('source') != 'native_worker_tensor_inventory'
            or record.get('model_config_sha256') != config_sha256(source_config)):
        raise ValueError('V4.1 serving requires source-matched native tensor and cache inventory')
    scope = record.get('scope')
    if not isinstance(scope, dict) or any(scope.get(key) != value for key, value in runtime.items()):
        raise ValueError('Native cache inventory does not match the simulator deployment')
    block, context, sequences = (scope.get(key) for key in
                                 ('block_size', 'max_model_len', 'max_num_seqs'))
    pages, stride, resident, arena = (record.get(key) for key in
        ('native_pages', 'page_stride_bytes', 'resident_gpu_bytes', 'arena_bytes'))
    if (any(type(value) is not int or value < 1 for value in
            (block, context, sequences, pages, stride, resident, arena))
            or pages*stride != arena
            or not isinstance(scope.get('kv_cache_dtype'), str)):
        raise ValueError('Native page-pool byte geometry is incomplete or inconsistent')
    groups = record.get('groups')
    if not isinstance(groups, list) or not groups:
        raise ValueError('Native cache groups are missing')
    variable, fixed = 0, 0
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError('Invalid native cache group')
        size, cap, kind = (group.get(key) for key in
                          ('block_size', 'max_blocks_per_request', 'kind'))
        if any(type(value) is not int or value < 1 for value in (size, cap)):
            raise ValueError('Native cache group requires positive block geometry')
        if kind == 'CircularBufferSpec':
            fixed += cap
        elif kind in ('SlidingWindowMLASpec', 'MLAAttentionSpec'):
            if block % size or cap != (context+size-1)//size:
                raise ValueError('Native full-history group is incompatible with the logical block')
            variable += block//size
        else:
            raise ValueError(f'Unsupported native cache group: {kind}')
    if not variable or not fixed:
        raise ValueError('Native V4.1 requires variable cache and fixed compressor state')
    # The shared block pool reserves block zero. Keep one fixed state span for
    # each possible running request before allocating the full-history pool.
    pool = NativePagePool(resident, pages, stride, block, variable,
                          1+fixed*sequences, context, sequences, scope['kv_cache_dtype'])
    if pool.logical_blocks < sequences*((context+block-1)//block):
        raise NotImplementedError('Native memory pressure needs group-aware cache scheduling')
    return pool


def validate_native_workload(path, pool, prefix_caching, request_limit=0):
    """Exclude cross-request reuse, which needs native bounded SWA replay."""
    if not path:
        raise ValueError('Native page-pool admission requires a bounded flat dataset')
    prefixes = set()
    count = 0
    with open(path) as stream:
        for line in stream:
            if request_limit > 0 and count >= request_limit:
                break
            row = json.loads(line)
            if 'sub_requests' in row:
                raise NotImplementedError('Native page-pool admission requires independent flat requests')
            prompt, output = row.get('input_toks'), row.get('output_toks')
            if (any(type(n) is not int or n < 1 for n in (prompt, output))
                    or prompt+output > pool.max_model_len):
                raise ValueError('Request exceeds the observed native context bound')
            if prefix_caching and prompt+output >= pool.block_size:
                ids = row.get('input_tok_ids')
                if not isinstance(ids, list) or len(ids) != prompt:
                    raise ValueError('Native prefix admission requires the full prompt token IDs')
                key = tuple(ids[:pool.block_size])
                if len(key) == pool.block_size and key in prefixes:
                    raise NotImplementedError('Reusable prefixes need native bounded SWA replay')
                prefixes.add(key)
            count += 1
    if not count:
        raise ValueError('Native page-pool admission requires at least one request')
    return count
