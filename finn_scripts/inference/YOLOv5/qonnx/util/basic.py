# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Xilinx nor the names of its contributors may be used to
#   endorse or promote products derived from this software without specific prior
#   written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import random
import string
import warnings

from qonnx.core.datatype import DataType

try:
    from onnx.helper import make_model, make_opsetid
except ModuleNotFoundError:
    make_model = None
    make_opsetid = None


def get_preferred_onnx_opset():
    return 11


def qonnx_make_model(graph_proto, **kwargs):
    opset_imports = kwargs.pop("opset_imports", None)
    if opset_imports is None:
        opset_imports = [make_opsetid("", get_preferred_onnx_opset())]
        kwargs["opset_imports"] = opset_imports
    else:
        kwargs["opset_imports"] = opset_imports
    return make_model(graph_proto, **kwargs)


def is_finn_op(op_type):
    return op_type.startswith("finn") or op_type.startswith("qonnx.custom_op") or op_type.startswith("onnx.brevitas")


def get_num_default_workers():
    try:
        return int(os.environ["NUM_DEFAULT_WORKERS"])
    except KeyError:
        return 1


def get_execution_error_thresh():
    try:
        return float(os.environ["ERROR_THRESH"])
    except KeyError:
        return 1e-2


def get_sanitize_quant_tensors():
    try:
        return int(os.environ["SANITIZE_QUANT_TENSORS"])
    except KeyError:
        return 1


def get_by_name(container, name, name_field="name"):
    names = [getattr(x, name_field) for x in container]
    inds = [i for i, e in enumerate(names) if e == name]
    if len(inds) > 1:
        raise Exception("Found multiple get_by_name matches, undefined behavior")
    elif len(inds) == 0:
        return None
    else:
        return container[inds[0]]


def remove_by_name(container, name, name_field="name"):
    item = get_by_name(container, name, name_field)
    if item is not None:
        container.remove(item)


def random_string(stringLength=6):
    lettersAndDigits = string.ascii_letters + string.digits
    return "".join(random.choice(lettersAndDigits) for i in range(stringLength))


def interleave_matrix_outer_dim_from_partitions(matrix, n_partitions):
    if type(matrix) != np.ndarray or matrix.dtype != np.float32:
        matrix = np.asarray(matrix, dtype=np.float32)
    shp = matrix.shape
    ndim = matrix.ndim
    assert shp[0] % n_partitions == 0
    assert ndim == 2
    matrix_r = matrix.reshape(-1, n_partitions, shp[1]).transpose((1, 0, 2))
    matrix_r = matrix_r.reshape(n_partitions, -1, shp[1])
    return matrix_r


def roundup_to_integer_multiple(x, factor):
    assert int(x) == x, "The input x is not an integer."
    assert int(factor) == factor, "The input factor is not an integer."
    if factor == -1:
        return x
    assert factor > 0 and x > 0, "Factor and x are <= 0."
    if x < factor:
        return factor
    else:
        if x % factor == 0:
            return x
        else:
            return x + (factor - (x % factor))


def pad_tensor_to_multiple_of(ndarray, pad_to_dims, val=0, distr_pad=False):
    if type(ndarray) != np.ndarray or ndarray.dtype != np.float32:
        ndarray = np.asarray(ndarray, dtype=np.float32)
    assert ndarray.ndim == len(pad_to_dims)
    desired = zip(list(ndarray.shape), list(pad_to_dims))
    desired = map(lambda x: roundup_to_integer_multiple(x[0], x[1]), desired)
    desired = np.asarray(list(desired), dtype=np.int32)
    current = np.asarray(ndarray.shape, dtype=np.int32)
    pad_amt = desired - current
    if distr_pad:
        pad_before = (pad_amt // 2).astype(np.int32)
        pad_after = pad_amt - pad_before
        pad_amt = list(zip(pad_before, pad_after))
    else:
        pad_amt = list(map(lambda x: (0, x), pad_amt))
    ret = np.pad(ndarray, pad_amt, mode="constant", constant_values=val)
    assert (np.asarray(ret.shape, dtype=np.int32) == desired).all()
    return ret


def calculate_matvec_accumulator_range(matrix: np.ndarray, vec_dt: DataType):
    max_vectors = np.where(matrix > 0, vec_dt.max(), vec_dt.min())
    min_vectors = np.where(matrix > 0, vec_dt.min(), vec_dt.max())
    max_value = (matrix * max_vectors).sum(axis=0).max()
    min_value = (matrix * min_vectors).sum(axis=0).min()
    return (min_value, max_value)


def gen_finn_dt_tensor(finn_dt, tensor_shape):
    if type(tensor_shape) == list:
        tensor_shape = tuple(tensor_shape)
    if finn_dt == DataType["BIPOLAR"]:
        tensor_values = np.random.randint(2, size=tensor_shape)
        tensor_values = 2 * tensor_values - 1
    elif finn_dt == DataType["BINARY"]:
        tensor_values = np.random.randint(2, size=tensor_shape)
    elif "INT" in finn_dt.name or finn_dt == DataType["TERNARY"]:
        tensor_values = np.random.randint(finn_dt.min(), high=finn_dt.max() + 1, size=tensor_shape)
    elif "FIXED" in finn_dt.name:
        int_dt = DataType["INT" + str(finn_dt.bitwidth())]
        tensor_values = np.random.randint(int_dt.min(), high=int_dt.max() + 1, size=tensor_shape)
        tensor_values = tensor_values * finn_dt.scale_factor()
    elif finn_dt == DataType["FLOAT32"]:
        tensor_values = np.random.randn(*tensor_shape)
    else:
        raise ValueError("Datatype {} is not supported, no tensor could be generated".format(finn_dt))
    return tensor_values.astype(np.float32)


def calculate_signed_dot_prod_range(dt_a, dt_b, len):
    assert dt_a.signed() and dt_b.signed()
    min_prod = 2**30
    max_prod = -(2**30)
    for a_val in [dt_a.min(), dt_a.max()]:
        for b_val in [dt_b.min(), dt_b.max()]:
            prod = a_val * b_val * len
            if prod < min_prod:
                min_prod = prod
            if prod > max_prod:
                max_prod = prod
    return (min_prod, max_prod)
