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
cimport cuvs.common.cydlpack

from cuvs.common.temp_raft import auto_sync_resources
from cuvs.common cimport cydlpack

from cython.operator cimport dereference as deref

from pylibraft.common import (
    DeviceResources,
    auto_convert_output,
    cai_wrapper,
    device_ndarray,
)
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from pylibraft.neighbors.common import _check_input_array
from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from pylibraft.common.handle cimport device_resources

from libc.stdint cimport int8_t, int64_t, uint8_t, uint32_t, uint64_t, uintptr_t


cdef class IndexParams:
    """
    Parameters to build index for CAGRA nearest neighbor search

    Parameters
    ----------
    metric : string denoting the metric type, default="sqeuclidean"
        Valid values for metric: ["sqeuclidean"], where
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2
    intermediate_graph_degree : int, default = 128

    graph_degree : int, default = 64

    build_algo: string denoting the graph building algorithm to use, \
                default = "ivf_pq"
        Valid values for algo: ["ivf_pq", "nn_descent"], where
            - ivf_pq will use the IVF-PQ algorithm for building the knn graph
            - nn_descent (experimental) will use the NN-Descent algorithm for
              building the knn graph. It is expected to be generally
              faster than ivf_pq.
    """
    cdef cagra_c.cagraIndexParams params

    def __init__(self, *,
                 metric="sqeuclidean",
                 intermediate_graph_degree=128,
                 graph_degree=64,
                 build_algo="ivf_pq",
                 nn_descent_niter=20):
        # todo (dgd): enable once other metrics are present
        # and exposed in cuVS C API
        # self.params.metric = _get_metric(metric)
        # self.params.metric_arg = 0
        self.params.intermediate_graph_degree = intermediate_graph_degree
        self.params.graph_degree = graph_degree
        if build_algo == "ivf_pq":
            self.params.build_algo = cagra_c.cagraGraphBuildAlgo.IVF_PQ
        elif build_algo == "nn_descent":
            self.params.build_algo = cagra_c.cagraGraphBuildAlgo.NN_DESCENT
        self.params.nn_descent_niter = nn_descent_niter

    # @property
    # def metric(self):
        # return self.params.metric

    @property
    def intermediate_graph_degree(self):
        return self.params.intermediate_graph_degree

    @property
    def graph_degree(self):
        return self.params.graph_degree

    @property
    def build_algo(self):
        return self.params.build_algo

    @property
    def nn_descent_niter(self):
        return self.params.nn_descent_niter


cdef class Index:
    cdef cagra_c.cagraIndex_t index

    def __cinit__(self):
        cdef cuvsError_t index_create_status
        index_create_status = cagra_c.cagraIndexCreate(&self.index)
        self.trained = False

        if index_create_status == cuvsError_t.CUVS_ERROR:
            raise Exception("FAIL")

    def __dealloc__(self):
        cdef cuvsError_t index_destroy_status
        if self.index is not NULL:
            index_destroy_status = cagra_c.cagraIndexDestroy(self.index)
            if index_destroy_status == cuvsError_t.CUVS_ERROR:
                raise Exception("FAIL")

    def __repr__(self):
        # todo(dgd): update repr as we expose data through C API
        attr_str = []
        return "Index(type=CAGRA, metric=L2" + (", ".join(attr_str)) + ")"


@auto_sync_resources
def build_index(IndexParams index_params, dataset, resources=None):
    """
    Build the CAGRA index from the dataset for efficient search.

    The build performs two different steps- first an intermediate knn-graph is
    constructed, then it's optimized it to create the final graph. The
    index_params object controls the node degree of these graphs.

    It is required that both the dataset and the optimized graph fit the
    GPU memory.

    The following distance metrics are supported:
        - L2

    Parameters
    ----------
    index_params : IndexParams object
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    {handle_docstring}

    Returns
    -------
    index: cuvs.cagra.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from pylibraft.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> handle = DeviceResources()
    >>> build_params = cagra.IndexParams(metric="sqeuclidean")
    >>> index = cagra.build_index(build_params, dataset)
    >>> distances, neighbors = cagra.search(cagra.SearchParams(),
    ...                                      index, dataset,
    ...                                      k, handle=handle)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    """

    # todo(dgd): we can make the check of dtype a parameter of wrap_array
    # in RAFT to make this a single call
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('byte'),
                                    np.dtype('ubyte')])

    if resources is None:
        resources = DeviceResources()
    cdef cuvsResources_t* resources_ = \
        <cuvsResources_t*><size_t>resources.getHandle()

    cdef Index idx = Index()
    cdef cuvsError_t build_status

    with cuda_interruptible():
        build_status = cagra_c.cagraBuild(
            deref(resources_),
            index_params.params,
            <cydlpack.DLManagedTensor*> &cydlpack(dataset_ai),
            deref(idx.index)
        )

        if build_status == cagra_c.cuvsError_t.CUVS_ERROR:
            raise RuntimeError("Index failed to build.")
        else:
            idx.trained = True

    return idx


cdef class SearchParams:
    """
    CAGRA search parameters

    Parameters
    ----------
    max_queries: int, default = 0
        Maximum number of queries to search at the same time (batch size).
        Auto select when 0.
    itopk_size: int, default = 64
        Number of intermediate search results retained during the search.
        This is the main knob to adjust trade off between accuracy and
        search speed. Higher values improve the search accuracy.
    max_iterations: int, default = 0
        Upper limit of search iterations. Auto select when 0.
    algo: string denoting the search algorithm to use, default = "auto"
        Valid values for algo: ["auto", "single_cta", "multi_cta"], where
            - auto will automatically select the best value based on query size
            - single_cta is better when query contains larger number of
              vectors (e.g >10)
            - multi_cta is better when query contains only a few vectors
    team_size: int, default = 0
        Number of threads used to calculate a single distance. 4, 8, 16,
        or 32.
    search_width: int, default = 1
        Number of graph nodes to select as the starting point for the
        search in each iteration.
    min_iterations: int, default = 0
        Lower limit of search iterations.
    thread_block_size: int, default = 0
        Thread block size. 0, 64, 128, 256, 512, 1024.
        Auto selection when 0.
    hashmap_mode: string denoting the type of hash map to use.
        It's usually better to allow the algorithm to select this value,
        default = "auto".
        Valid values for hashmap_mode: ["auto", "small", "hash"], where
            - auto will automatically select the best value based on algo
            - small will use the small shared memory hash table with resetting.
            - hash will use a single hash table in global memory.
    hashmap_min_bitlen: int, default = 0
        Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    hashmap_max_fill_rate: float, default = 0.5
        Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    num_random_samplings: int, default = 1
        Number of iterations of initial random seed node selection. 1 or
        more.
    rand_xor_mask: int, default = 0x128394
        Bit mask used for initial random seed node selection.
    """
    cdef cagra_c.cagraSearchParams params

    def __init__(self, *,
                 max_queries=0,
                 itopk_size=64,
                 max_iterations=0,
                 algo="auto",
                 team_size=0,
                 search_width=1,
                 min_iterations=0,
                 thread_block_size=0,
                 hashmap_mode="auto",
                 hashmap_min_bitlen=0,
                 hashmap_max_fill_rate=0.5,
                 num_random_samplings=1,
                 rand_xor_mask=0x128394):
        self.params.max_queries = max_queries
        self.params.itopk_size = itopk_size
        self.params.max_iterations = max_iterations
        if algo == "single_cta":
            self.params.algo = cagra_c.cagraSearchAlgo.SINGLE_CTA
        elif algo == "multi_cta":
            self.params.algo = cagra_c.cagraSearchAlgo.MULTI_CTA
        elif algo == "multi_kernel":
            self.params.algo = cagra_c.cagraSearchAlgo.MULTI_KERNEL
        elif algo == "auto":
            self.params.algo = cagra_c.cagraSearchAlgo.AUTO
        else:
            raise ValueError("`algo` value not supported.")

        self.params.team_size = team_size
        self.params.search_width = search_width
        self.params.min_iterations = min_iterations
        self.params.thread_block_size = thread_block_size
        if hashmap_mode == "hash":
            self.params.hashmap_mode = cagra_c.cagraHashMode.HASH
        elif hashmap_mode == "small":
            self.params.hashmap_mode = cagra_c.cagraHashMode.SMALL
        elif hashmap_mode == "auto":
            self.params.hashmap_mode = cagra_c.cagraHashMode.AUTO_HASH
        else:
            raise ValueError("`hashmap_mode` value not supported.")

        self.params.hashmap_min_bitlen = hashmap_min_bitlen
        self.params.hashmap_max_fill_rate = hashmap_max_fill_rate
        self.params.num_random_samplings = num_random_samplings
        self.params.rand_xor_mask = rand_xor_mask

    def __repr__(self):
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in [
                        "max_queries", "itopk_size", "max_iterations", "algo",
                        "team_size", "search_width", "min_iterations",
                        "thread_block_size", "hashmap_mode",
                        "hashmap_min_bitlen", "hashmap_max_fill_rate",
                        "num_random_samplings", "rand_xor_mask"]]
        return "SearchParams(type=CAGRA, " + (", ".join(attr_str)) + ")"

    @property
    def max_queries(self):
        return self.params.max_queries

    @property
    def itopk_size(self):
        return self.params.itopk_size

    @property
    def max_iterations(self):
        return self.params.max_iterations

    @property
    def algo(self):
        return self.params.algo

    @property
    def team_size(self):
        return self.params.team_size

    @property
    def search_width(self):
        return self.params.search_width

    @property
    def min_iterations(self):
        return self.params.min_iterations

    @property
    def thread_block_size(self):
        return self.params.thread_block_size

    @property
    def hashmap_mode(self):
        return self.params.hashmap_mode

    @property
    def hashmap_min_bitlen(self):
        return self.params.hashmap_min_bitlen

    @property
    def hashmap_max_fill_rate(self):
        return self.params.hashmap_max_fill_rate

    @property
    def num_random_samplings(self):
        return self.params.num_random_samplings

    @property
    def rand_xor_mask(self):
        return self.params.rand_xor_mask


@auto_sync_resources
@auto_convert_output
def search(SearchParams search_params,
           Index index,
           queries,
           k,
           neighbors=None,
           distances=None,
           resources=None):
    """
    Find the k nearest neighbors for each query.

    Parameters
    ----------
    search_params : SearchParams
    index : Index
        Trained CAGRA index.
    queries : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    k : int
        The number of neighbors.
    neighbors : Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional CUDA array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    {handle_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = cagra.build(cagra.IndexParams(), dataset, handle=handle)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = cagra.SearchParams(
    ...     max_queries=100,
    ...     itopk_size=64
    ... )
    >>> # Using a pooling allocator reduces overhead of temporary array
    >>> # creation during search. This is useful if multiple searches
    >>> # are performad with same query size.
    >>> distances, neighbors = cagra.search(search_params, index, queries,
    ...                                     k, handle=handle)
    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()
    >>> neighbors = cp.asarray(neighbors)
    >>> distances = cp.asarray(distances)
    """
    if not index.trained:
        raise ValueError("Index need to be built before calling search.")

    if resources is None:
        resources = DeviceResources()
    cdef device_resources* resources_ = \
        <device_resources*><size_t>resources.getHandle()

    # todo(dgd): we can make the check of dtype a parameter of wrap_array
    # in RAFT to make this a single call
    queries_cai = cai_wrapper(queries)
    _check_input_array(queries_cai, [np.dtype('float32'), np.dtype('byte'),
                                     np.dtype('ubyte')],
                       exp_cols=index.dim)

    cdef uint32_t n_queries = queries_cai.shape[0]

    if neighbors is None:
        neighbors = device_ndarray.empty((n_queries, k), dtype='uint32')

    neighbors_cai = cai_wrapper(neighbors)
    _check_input_array(neighbors_cai, [np.dtype('uint32')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    distances_cai = cai_wrapper(distances)
    _check_input_array(distances_cai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    cdef cagra_c.cagraSearchParams params = search_params.params

    with cuda_interruptible():
        cagra_c.cagraSearch(
            <cuvsResources_t> resources_,
            params,
            idx_float.index,
            <cydlpack.DLManagedTensor*> cydlpack(queries_cai),
            <cydlpack.DLManagedTensor*> cydlpack(neighbors_cai),
            <cydlpack.DLManagedTensor*> cydlpack(distances_cai)
        )

    return (distances, neighbors)
