# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
from typing import Type
import itertools

import numpy as np

from sparta.specializer import kernels


SHAPE_CONFIG = {
    'GLOBAL_M_VALUE': 1024,
    'GLOBAL_K_VALUE': 256,
    'GLOBAL_N_VALUE': 512,
}
TILE_CONFIG = {
    'BLOCK_SIZE_M_VALUE': 64,
    'BLOCK_SIZE_K_VALUE': 32,
    'BLOCK_SIZE_N_VALUE': 128,
    'THREAD_SIZE_M_VALUE': 8,
    'THREAD_SIZE_K_VALUE': 4,
    'THREAD_SIZE_N_VALUE': 16,
}


def test_matmul_kernel(kernel_class: Type[kernels.KernelBase], s, b, t, c, cfg):
    np.random.seed(2022)
    kernel = kernel_class(sparse_type=s, biased=b, transpose=t, compressed=c)
    # test_pkg_dir = (f"./{kernel.get_kernel_name()}_"
    #                              f"m{cfg['GLOBAL_M_VALUE']}_"
    #                              f"k{cfg['GLOBAL_K_VALUE']}_"
    #                              f"n{cfg['GLOBAL_N_VALUE']}_"
    #                              f"bm{cfg['BLOCK_SIZE_M_VALUE']}_"
    #                              f"bk{cfg['BLOCK_SIZE_K_VALUE']}_"
    #                              f"bn{cfg['BLOCK_SIZE_N_VALUE']}_"
    #                              f"tm{cfg['THREAD_SIZE_M_VALUE']}_"
    #                              f"tk{cfg['THREAD_SIZE_K_VALUE']}_"
    #                              f"tn{cfg['THREAD_SIZE_N_VALUE']}")
    kernel_filepath = (f"./{kernel.get_kernel_name()}_"
                                 f"m{cfg['GLOBAL_M_VALUE']}_"
                                 f"k{cfg['GLOBAL_K_VALUE']}_"
                                 f"n{cfg['GLOBAL_N_VALUE']}_"
                                 f"bm{cfg['BLOCK_SIZE_M_VALUE']}_"
                                 f"bk{cfg['BLOCK_SIZE_K_VALUE']}_"
                                 f"bn{cfg['BLOCK_SIZE_N_VALUE']}_"
                                 f"tm{cfg['THREAD_SIZE_M_VALUE']}_"
                                 f"tk{cfg['THREAD_SIZE_K_VALUE']}_"
                                 f"tn{cfg['THREAD_SIZE_N_VALUE']}"
                                 f".cu")
    # print(f"Rendering: {kernel_filepath}")
    kernel.render_test_code(cfg, kernel_filepath)
    # kernel.render_test_pkg(cfg, num_iters=1000, dir=test_pkg_dir)
    try:
        latency = kernel.test(cfg, num_iters=1000)
    except Exception as e:
        print(e)
        latency = None
    print(f'{kernel_filepath}: {latency} ms')
    return kernel_filepath, latency


def build_matmul_kernel_pkg(kernel_class: Type[kernels.KernelBase], s, b, t, c, cfg):
    np.random.seed(2022)
    kernel = kernel_class(sparse_type=s, biased=b, transpose=t, compressed=c)
    test_pkg_dir = (f"./{kernel.get_kernel_name()}_"
                                 f"m{cfg['GLOBAL_M_VALUE']}_"
                                 f"k{cfg['GLOBAL_K_VALUE']}_"
                                 f"n{cfg['GLOBAL_N_VALUE']}_"
                                 f"bm{cfg['BLOCK_SIZE_M_VALUE']}_"
                                 f"bk{cfg['BLOCK_SIZE_K_VALUE']}_"
                                 f"bn{cfg['BLOCK_SIZE_N_VALUE']}_"
                                 f"tm{cfg['THREAD_SIZE_M_VALUE']}_"
                                 f"tk{cfg['THREAD_SIZE_K_VALUE']}_"
                                 f"tn{cfg['THREAD_SIZE_N_VALUE']}")
    print(f"Rendering: {test_pkg_dir}")
    kernel.render_test_pkg(cfg, num_iters=1000, dir=test_pkg_dir)
    return test_pkg_dir


def run_sparta_sparse_matmul_sweep():
    print('==================== Testing SparTA Sparse Matmul Kernels ====================')

    shape_config_parameters = {
        'GLOBAL_M_VALUE': [1024],
        'GLOBAL_K_VALUE': [256],
        'GLOBAL_N_VALUE': [512],
    }
    shape_param_keys, shape_param_values = zip(*shape_config_parameters.items())

    kernel_parameters = {
        'BLOCK_SIZE_M_VALUE': [16, 32, 64],
        'BLOCK_SIZE_N_VALUE': [16, 32, 64],
        'BLOCK_SIZE_K_VALUE': [16, 32, 64],
        'THREAD_SIZE_M_VALUE': [4, 8],
        'THREAD_SIZE_N_VALUE': [4, 8],
        'THREAD_SIZE_K_VALUE': [4, 8],
    }
    kernel_param_keys, kernel_param_values = zip(*kernel_parameters.items())

    results = {}
    for stype in ['sdd',]: # 'dsd', 'dds']:
        for biased in [False, True]:
            for transpose in [False, True]:
                for compressed in [False, True]:
                    for shape_config in [dict(zip(shape_param_keys, v)) for v in itertools.product(*shape_param_values)]:
                        for tile_config in [dict(zip(kernel_param_keys, v)) for v in itertools.product(*kernel_param_values)]:
                            build_matmul_kernel_pkg(
                                kernels.SparTATemplateSparseMatMulKernel,
                                stype, biased, transpose, compressed,
                                dict(shape_config, **tile_config)
                            )
                            # kernel_filepath, latency = test_matmul_kernel(
                            #     kernels.SparTATemplateSparseMatMulKernel,
                            #     stype, biased, transpose, compressed,
                            #     dict(shape_config, **tile_config)
                            # )
                            # results[kernel_filepath] = latency

    import csv

    with open('benchmark_results.csv', 'w') as fp_csv:
        writer = csv.writer(fp_csv)
        writer.writerow(['kernel_filepath', 'latency_ms'])
        for key, value in results.items():
            writer.writerow([key, value])


def build_openai_matmul_kernel_pkg(kernel_class: Type[kernels.KernelBase], s, b, t, c, cfg):
    np.random.seed(2022)
    kernel = kernel_class(sparse_type=s, biased=b, transpose=t, compressed=c)
    test_pkg_dir = (f"./{kernel.get_kernel_name()}_"
                                 f"m{cfg['GLOBAL_M_VALUE']}_"
                                 f"k{cfg['GLOBAL_K_VALUE']}_"
                                 f"n{cfg['GLOBAL_N_VALUE']}_"
                                 f"bm{cfg['BLOCK_SIZE_M_VALUE']}_"
                                 f"bk{cfg['BLOCK_SIZE_K_VALUE']}_"
                                 f"bn{cfg['BLOCK_SIZE_N_VALUE']}_"
                                 f"tm{cfg['THREAD_SIZE_M_VALUE']}_"
                                 f"tk{cfg['THREAD_SIZE_K_VALUE']}_"
                                 f"tn{cfg['THREAD_SIZE_N_VALUE']}")
    print(f"Rendering: {test_pkg_dir}")
    kernel.render_test_pkg(cfg, num_iters=1000, dir=test_pkg_dir)
    return test_pkg_dir


def run_openai_sparse_matmul_sweep():
    print('==================== Testing SparTA Sparse Matmul Kernels ====================')

    shape_config_parameters = {
        'GLOBAL_M_VALUE': [1024],
        'GLOBAL_K_VALUE': [256],
        'GLOBAL_N_VALUE': [512],
    }
    shape_param_keys, shape_param_values = zip(*shape_config_parameters.items())

    kernel_parameters = {
        'BLOCK_SIZE_M_VALUE': [16, 32, 64],
        'BLOCK_SIZE_N_VALUE': [16, 32, 64],
        'BLOCK_SIZE_K_VALUE': [16, 32, 64],
        'THREAD_SIZE_M_VALUE': [4, 8],
        'THREAD_SIZE_N_VALUE': [4, 8],
        'THREAD_SIZE_K_VALUE': [4, 8],
    }
    kernel_param_keys, kernel_param_values = zip(*kernel_parameters.items())

    results = {}
    for stype in ['sdd',]: # 'dsd', 'dds']:
        for biased in [False, True]:
            for transpose in [False, True]:
                for compressed in [False, True]:
                    for shape_config in [dict(zip(shape_param_keys, v)) for v in itertools.product(*shape_param_values)]:
                        for tile_config in [dict(zip(kernel_param_keys, v)) for v in itertools.product(*kernel_param_values)]:
                            build_matmul_kernel_pkg(
                                kernels.OpenAITemplateSparseMatMulKernel,
                                stype, biased, transpose, compressed,
                                dict(shape_config, **tile_config)
                            )
                            # kernel_filepath, latency = test_matmul_kernel(
                            #     kernels.SparTATemplateSparseMatMulKernel,
                            #     stype, biased, transpose, compressed,
                            #     dict(shape_config, **tile_config)
                            # )
                            # results[kernel_filepath] = latency

    import csv

    with open('benchmark_results.csv', 'w') as fp_csv:
        writer = csv.writer(fp_csv)
        writer.writerow(['kernel_filepath', 'latency_ms'])
        for key, value in results.items():
            writer.writerow([key, value])


if __name__ == '__main__':
    # run_sparta_sparse_matmul_sweep()
    run_openai_sparse_matmul_sweep()