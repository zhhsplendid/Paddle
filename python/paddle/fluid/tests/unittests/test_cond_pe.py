#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import os
import unittest

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import Program, program_guard
from simple_nets import simple_fc_net_with_inputs, batchnorm_fc_with_inputs
from test_cond import TestCondBackward

np.random.seed(123)



class TestCondBackwardWithParallelExecutor(TestCondBackward):
    def test_cond_backward(self):
        def cond_func(i, img, label):
            predicate = ((i % 2) == 0)
            return layers.cond(predicate,
                               lambda: simple_fc_net_with_inputs(img, label, class_num=10),
                               lambda: batchnorm_fc_with_inputs(img, label, class_num=10))

        
        self.backward_value_helper(cond_func,
                                       core.is_compiled_with_cuda(),
                                       use_parallel_exe=True)
        self.add_optimizer_helper(cond_func,
                                      core.is_compiled_with_cuda(),
                                      use_parallel_exe=True)

    def test_half_nested_cond_backward(self):
        def branch(i, img, label):
            return layers.cond((i % 2) == 0, lambda: simple_fc_net_with_inputs(img, label, class_num=10),
                               lambda: batchnorm_fc_with_inputs(img, label, class_num=10))

        def cond_func_simple_net_at_true(i, img, label):
            return layers.cond(i < 5, lambda: branch(i, img, label),
                               lambda: layers.mean(img))

        def cond_func_simple_net_at_false(i, img, label):
            return layers.cond(i < 5, lambda: layers.mean(img),
                               lambda: branch(i, img, label))

        
        self.backward_value_helper(cond_func_simple_net_at_true,
                                       core.is_compiled_with_cuda(),
                                       use_parallel_exe=True)
        self.add_optimizer_helper(cond_func_simple_net_at_true,
                                      core.is_compiled_with_cuda(),
                                      use_parallel_exe=True)
        self.backward_value_helper(cond_func_simple_net_at_false,
                                       core.is_compiled_with_cuda(),
                                       use_parallel_exe=True)
        self.add_optimizer_helper(cond_func_simple_net_at_false,
                                      core.is_compiled_with_cuda(),
                                      use_parallel_exe=True)

    def test_nested_cond_backward(self):
        def branch(i, img, label, mod_two):

            if mod_two:
                predicate = ((i % 2) == 0)
            else:
                predicate = ((i % 2) != 0)
            return layers.cond(predicate, lambda: simple_fc_net_with_inputs(img, label, class_num=10),
                               lambda: batchnorm_fc_with_inputs(img, label, class_num=10))

        def cond_func(i, img, label):
            return layers.cond(i < 5, lambda: branch(i, img, label, True),
                               lambda: branch(i, img, label, False))

        
        self.backward_value_helper(cond_func,
                                       core.is_compiled_with_cuda(),
                                       use_parallel_exe=True)
        self.add_optimizer_helper(cond_func,
                                      core.is_compiled_with_cuda(),
                                      use_parallel_exe=True)


if __name__ == '__main__':
    unittest.main()
