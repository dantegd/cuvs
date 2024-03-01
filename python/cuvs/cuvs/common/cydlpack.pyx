#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cython: language_level=3

import numpy as np
# from libc.stdio cimport printf

from libc cimport stdlib

from libc.stdint cimport uintptr_t


cdef void deleter(DLManagedTensor* tensor) noexcept:
    if tensor.manager_ctx is NULL:
        return
    stdlib.free(tensor.dl_tensor.shape)
    tensor.manager_ctx = NULL
    stdlib.free(tensor)


cdef DLManagedTensor* dlpack_c(ary):
    # todo(dgd): add checking options/parameters
    cdef DLDeviceType dev_type
    cdef DLDevice dev
    cdef DLDataType dtype
    cdef DLTensor tensor
    cdef DLManagedTensor* dlm = \
        <DLManagedTensor*>stdlib.malloc(sizeof(DLManagedTensor))

    if ary.from_cai:
        print("A")
        dev_type = DLDeviceType.kDLCUDA
    else:
        print("B")
        dev_type = DLDeviceType.kDLCPU

    dev.device_type = dev_type
    dev.device_id = 0

    # todo (dgd): change to nice dict
    if ary.dtype == np.float32:
        dtype.code = DLDataTypeCode.kDLFloat
        dtype.bits = 32
        print("dtype 1: ", DLDataTypeCode.kDLFloat, 32)
    elif ary.dtype == np.float64:
        dtype.code = DLDataTypeCode.kDLFloat
        dtype.bits = 64
        print("dtype 1: ", DLDataTypeCode.kDLFloat, 64)
    elif ary.dtype == np.int32:
        dtype.code = DLDataTypeCode.kDLInt
        dtype.bits = 32
        print("dtype 1: ", DLDataTypeCode.kDLInt, 32)
    elif ary.dtype == np.int64:
        dtype.code = DLDataTypeCode.kDLInt
        dtype.bits = 64
        print("dtype 1: ", DLDataTypeCode.kDLFloat, 64)
    elif ary.dtype == np.uint32:
        dtype.code = DLDataTypeCode.kDLUInt
        dtype.bits = 32
        print("dtype 1: ", DLDataTypeCode.kDLInt, 32)
    elif ary.dtype == np.uint64:
        dtype.code = DLDataTypeCode.kDLUInt
        dtype.bits = 64
        print("dtype 1: ", DLDataTypeCode.kDLFloat, 64)
    elif ary.dtype == np.bool_:
        dtype.code = DLDataTypeCode.kDLFloat
        dtype.bits = 8
        print("dtype 1: ", DLDataTypeCode.kDLFloat, 16)

    dtype.lanes = 1



    cdef size_t ndim = len(ary.shape)

    cdef int64_t* shape = <int64_t*>stdlib.malloc(ndim * sizeof(int64_t))

    for i in range(ndim):
        shape[i] = ary.shape[i]

    print(ndim, ary.shape)

    tensor_ptr = ary.ai["data"][0]
    print("@@@@@: ", ary.ai_)

    tensor.data = <void*> tensor_ptr
    tensor.device = dev
    tensor.dtype = dtype
    tensor.strides = NULL
    tensor.ndim = ndim
    tensor.shape = shape
    tensor.byte_offset = 0

    print("tensor.data: ", ary.data)
    # print("C++ tensor.data: ", tensor.data)
    print("tensor.device: ", dev)
    print("tensor.dtype: ", dtype)
    print("tensor.strides: ", "NULL")
    print("tensor.ndim: ", ndim)
    print("tensor.shape: ", ary.shape)
    print("tensor.byte_offset: ", 0)


    dlm.dl_tensor = tensor
    dlm.manager_ctx = NULL
    dlm.deleter = deleter

    # printf("%p\n", tensor.data)

    return dlm
