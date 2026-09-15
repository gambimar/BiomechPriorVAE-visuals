"""Load Generative GaitNet's released checkpoints on modern Python.

The released `data/trained_nn/*` files were pickled by Ray 1.8 under Python 3.6.
Ray's cloudpickle embeds *code objects* for the policy classes, and `CodeType`
gained `posonlyargcount` in Python 3.8, so a plain `pickle.loads` dies with
`TypeError: an integer is required (got type bytes)` the moment it reaches one.

None of that code is needed to run a policy. The only things required are:

    worker["filters"]["default_policy"]   observation filter (NoFilter here)
    worker["state"]["default_policy"]     the network weights (numpy arrays)

so this unpickler replaces the class/code reconstruction machinery with inert
stubs and lets the plain data through untouched. Numpy arrays, dtypes and dicts
are still unpickled normally, so the weights are exact -- nothing is
approximated.
"""
import io
import pickle

import torch
import torch.storage

# The checkpoints were saved from GPU workers, so the embedded torch storages
# carry a CUDA location tag. Nested `torch.load` calls inside the pickle stream
# get no map_location argument, so they would demand a GPU (or land tensors on
# one while the rest of the rollout runs on CPU). Route them to the CPU.
_orig_load_from_bytes = torch.storage._load_from_bytes


def _load_from_bytes_cpu(b):
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)


torch.storage._load_from_bytes = _load_from_bytes_cpu


class _Stub:
    """Inert stand-in for anything that cannot be rebuilt on this interpreter.

    Must be a class, not a factory function: the pickle stream uses NEWOBJ for
    some of these, and NEWOBJ requires a type object.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __call__(self, *args, **kwargs):
        return _Stub(*args, **kwargs)

    def __setstate__(self, state):
        self._state = state

    def __reduce__(self):
        return (_Stub, ())


def _stub_factory(*args, **kwargs):
    return _Stub(*args, **kwargs)


# Modules whose contents are code/class machinery rather than data.
_STUB_MODULES = ('ray.cloudpickle', 'cloudpickle')


class GaitNetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if any(module.startswith(m) for m in _STUB_MODULES):
            return _Stub
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


def loads(data):
    return GaitNetUnpickler(io.BytesIO(data)).load()


def load_checkpoint_raw(path):
    """Return the outer checkpoint dict (worker state left as raw bytes)."""
    with open(path, 'rb') as fh:
        return GaitNetUnpickler(fh).load()


def load_policy_parts(path):
    """Return (policy_weights, filter_obj, cascading_type, muscle_state, metadata)."""
    state = load_checkpoint_raw(path)
    worker = loads(state['worker'])

    policy_state = worker['state']['default_policy']
    filter_state = worker['filters']['default_policy']

    if isinstance(policy_state, dict) and 'weights' in policy_state:
        weights = policy_state['weights']
    else:
        weights = policy_state
    if isinstance(weights, dict):
        weights = {k: v for k, v in weights.items() if k != '_optimizer_variables'}

    return (weights, filter_state, state.get('cascading_type', 0),
            state.get('muscle'), state.get('metadata'))


if __name__ == '__main__':
    import sys

    import numpy as np

    for path in sys.argv[1:]:
        w, f, ct, muscle, md = load_policy_parts(path)
        print('===', path)
        print('  cascading_type:', ct)
        print('  filter:', type(f).__name__, f)
        print('  weight keys:', list(w.keys())[:8], '...' if len(w) > 8 else '')
        k0 = list(w.keys())[0]
        print('  %s -> %s %s' % (k0, np.asarray(w[k0]).shape, np.asarray(w[k0]).dtype))
        print('  muscle keys:', list(muscle.keys())[:5] if isinstance(muscle, dict) else type(muscle))
