"""Checked inference-only batch adapter used by the corrected R0 worker."""
import ast,copy,difflib,functools,hashlib,inspect,textwrap
from pathlib import Path
_ORIGINALS={}
_POLICY={'s3':{'catalog_user_batch':32,'catalog_item_batch':8192,'candidate_group_batch_size':2048}}

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _original(owner, name):
    key = (id(owner), name)
    if key not in _ORIGINALS:
        _ORIGINALS[key] = getattr(owner, name)
    return _ORIGINALS[key]

def batch_only_ast(function, replacements, expected_counts):
    """Compile the identical source AST with only checked integer batch literals.

    The caller supplies a narrowly inspected function. Every replacement must
    lie in a range() step or a batch-boundary addition, never a model setting.
    The complete inverse-transformed AST must equal the original AST.
    """
    raw = inspect.unwrap(function)
    source = textwrap.dedent(inspect.getsource(raw))
    tree = ast.parse(source)
    baseline = copy.deepcopy(tree)
    counts = {k: 0 for k in replacements}
    changes = []

    class Replace(ast.NodeTransformer):
        def __init__(self):
            self.parents = []

        def visit(self, node):
            self.parents.append(node)
            result = super().visit(node)
            self.parents.pop()
            return result

        def visit_Constant(self, node):
            if isinstance(node.value, bool) or not isinstance(node.value, int) or node.value not in replacements:
                return node
            parent = self.parents[-2]
            range_step = (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                          and parent.func.id == 'range' and len(parent.args) == 3 and parent.args[2] is node)
            boundary = isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add) and parent.right is node
            if not (range_step or boundary):
                raise ValueError('Batch literal appears outside an approved loop/boundary: ' + ast.dump(parent))
            old = node.value
            counts[old] += 1
            changes.append(dict(line=node.lineno, column=node.col_offset,
                                old=old, new=int(replacements[old]),
                                role='range_step' if range_step else 'batch_boundary'))
            node.value = int(replacements[old])
            return node

    updated = Replace().visit(tree)
    if counts != expected_counts:
        raise ValueError(f'Unexpected batch literal count in {raw.__name__}: {counts} != {expected_counts}')
    reverted = copy.deepcopy(updated)
    positions = {(c['line'], c['column']): c['old'] for c in changes}
    for node in ast.walk(reverted):
        if isinstance(node, ast.Constant) and (getattr(node, 'lineno', None), getattr(node, 'col_offset', None)) in positions:
            node.value = positions[node.lineno, node.col_offset]
    if ast.dump(reverted, include_attributes=False) != ast.dump(baseline, include_attributes=False):
        raise AssertionError('A non-batch AST node changed')
    ast.fix_missing_locations(updated)
    namespace = {}
    exec(compile(updated, inspect.getsourcefile(raw), 'exec'), raw.__globals__, namespace)
    result = namespace[raw.__name__]
    result.__module__ = raw.__module__
    result.__qualname__ = raw.__qualname__
    old_text, new_text = ast.unparse(baseline) + '\n', ast.unparse(updated) + '\n'
    record = dict(function=raw.__module__ + '.' + raw.__qualname__,
                  source_file=inspect.getsourcefile(raw), source_sha256=_sha(inspect.getsourcefile(raw)),
                  original_function_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  only_batch_literals_changed=True, changes=changes,
                  normalized_source_diff=''.join(difflib.unified_diff(
                      old_text.splitlines(True), new_text.splitlines(True),
                      fromfile=raw.__name__ + ':original', tofile=raw.__name__ + ':batch-policy')))
    return result, record

def _patch_candidates(modules, records):
    config = _POLICY['s3']
    candidates, features, uniform = modules['candidates'], modules['features'], modules['uniform']
    catalog, record = batch_only_ast(_original(candidates, 'score_catalog'),
                                    {32: config['catalog_user_batch'], 2048: config['catalog_item_batch']},
                                    {32: 3, 2048: 2})
    candidates.score_catalog = catalog; records.append(record)
    statistics, record = batch_only_ast(_original(features, '_pair_statistics'),
                                       {8192: config['candidate_group_batch_size'] * 50}, {8192: 2})
    features._pair_statistics = statistics
    uniform._pair_statistics = statistics
    records.append(record)
    native, record = batch_only_ast(_original(features, 'native_scores'),
                                   {128: config['candidate_group_batch_size']}, {128: 2})
    features.native_scores = native
    features._old.native_scores = native
    records.append(record)
    compute = _original(features, '_compute_records')

    @functools.wraps(compute)
    def compute_large(*args, **kwargs):
        if len(args) > 5:
            args = list(args); args[5] = config['candidate_group_batch_size']; args = tuple(args)
            kwargs.pop('chunk_groups', None)
        else:
            kwargs['chunk_groups'] = config['candidate_group_batch_size']
        return compute(*args, **kwargs)

    features._compute_records = compute_large
