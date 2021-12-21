/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/gather_scatter_kernel.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class TensorAssign {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data;
    VLOG(3) << "self_data assigned:" << *self_data;
  }
};
static TensorAssign tensor_assign;

class ReduceAdd {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data += *src_data;
    VLOG(3) << "self_data added:" << *self_data;
  }
};

static ReduceAdd reduce_add;

class ReduceMultiply {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data *= *src_data;
  }

  void operator()(bool* self_data, bool* src_data) const {
    *self_data = *self_data && *src_data;
  }
};

static ReduceMultiply reduce_mul;

template <typename tensor_t, typename index_t = int64_t,
          bool is_scatter_like = true>
struct cpu_gather_scatter_functor {
  template <typename func_t>
  void operator()(Tensor self, int dim, const Tensor& index, const Tensor& src,
                  const std::string& method_name, const func_t& reduce_op,
                  const platform::DeviceContext& ctx) {
    if (index.numel() == 0) {
      return;
    }
    auto* self_data = self.data<tensor_t>();  // problem occour here
    VLOG(3) << "self_data:" << *self_data;
    auto* index_data = index.data<index_t>();
    VLOG(3) << "index_data:" << *index_data;
    auto* src_data = src.data<tensor_t>();
    int64_t self_size = self.numel();
    int64_t index_size = index.numel();
    int64_t src_size = src.numel();
    auto self_dims = self.dims();
    VLOG(3) << "index_size" << index_size;
    VLOG(3) << "self_size" << self_size;
    auto index_dims = index.dims();
    auto src_dims = src.dims();
    if (self_size == 0 || src_size == 0 || index_size == 0) {
      VLOG(3) << "zero size input found";
      platform::errors::InvalidArgument(
          "self_size, src_size, index_size cannot be 0");
      return;
    }
    int select_dim_size = index_dims[dim];
    // index matrix has different shape with self matrix or src matrix.
    int replaced_select_dim_size =
        is_scatter_like ? self_dims[dim] : src_dims[dim];
    int64_t inner_dim_size = 1;
    int64_t outer_dim_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
      inner_dim_size *= index_dims[i];
    }

    for (int i = dim + 1; i < index_dims.size(); i++) {
      outer_dim_size *= index_dims[i];
    }
    // Debug function
    // for (int n = 0; n < self_size; ++n) {
    //   VLOG(3) << "self value before scatter/gather:" << *(self_data + n);
    // }
    // VLOG(3) << "inner_dim_size:" << inner_dim_size;
    // VLOG(3) << "select_dim_size:" << select_dim_size;
    // VLOG(3) << "outer_dim_size:" << outer_dim_size;
    // VLOG(3) << "index_size:" << outer_dim_size;

    int64_t index_idx = 0;
    int64_t self_idx, src_idx;

    // N layer loop squeezed into 3 layers loop
    for (int64_t i = 0; i < inner_dim_size; i++) {
      for (int64_t j = 0; j < select_dim_size; j++) {
        for (int64_t k = 0; k < outer_dim_size; k++) {
          int64_t index = index_data[index_idx];

          /*
            gather computation formula:

            self[i][j][k] = src[index[i][j][k]][j][k]  # if dim == 0
            self[i][j][k] = src[i][index[i][j][k]][k]  # if dim == 1
            self[i][j][k] = src[i][j][index[i][j][k]]  # if dim == 2

            scatter computation formula:

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

          */

          // This index might out of bound of index matrix's index, so here
          // multiply the replaced_select_dim_size.
          int64_t replace_index = k + index * outer_dim_size +
                                  i * outer_dim_size * replaced_select_dim_size;

          self_idx = is_scatter_like ? replace_index : index_idx;
          src_idx = is_scatter_like ? index_idx : replace_index;

          reduce_op((tensor_t*)(self_data + self_idx),
                    (tensor_t*)(src_data + src_idx));
          index_idx++;

          // for debug
          // VLOG(3) << ">>>>>> i j k :" << i << j << k;
          // VLOG(3) << ">>>>>> index data:" << index;
          // VLOG(3) << ">>>>>> index_idx:" << index_idx;
          // VLOG(3) << ">>>>>> replace_index:" << replace_index;
        }
      }
    }

    // Debug function
    // for (int n = 0; n < self_size; ++n) {
    //   VLOG(3) << "self value after scatter:" << *(self_data + n);
    // }
  }
};

template <typename tensor_t, typename index_t>
void cpu_gather_kernel(Tensor self, int dim, const Tensor& index, Tensor result,
                       const platform::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t, index_t,
                             /*is_scatter_like=*/false>()(
      result, dim, index, self, "gather_out_cpu", tensor_assign, ctx);
  VLOG(3) << "<<<< Done cpu_gather_kernel <<<<<";
}

template <typename tensor_t, typename index_t>
void cpu_scatter_assign_kernel(Tensor self, int dim, const Tensor& index,
                               Tensor src, const platform::DeviceContext& ctx) {
  VLOG(3) << "start scatter assign kernel";
  VLOG(3) << "src data scatter:" << *(src.data<tensor_t>());
  cpu_gather_scatter_functor<tensor_t, index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_assign_cpu", tensor_assign, ctx);
  VLOG(3) << "<<<< Done cpu_scatter_assign_kernel <<<<<";
}

template <typename tensor_t, typename index_t>
void cpu_scatter_add_kernel(Tensor self, int dim, const Tensor& index,
                            Tensor src, const platform::DeviceContext& ctx) {
  VLOG(3) << "start scatter add kernel";
  VLOG(3) << "src data scatter:" << *(src.data<tensor_t>());
  cpu_gather_scatter_functor<tensor_t, index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_add_cpu", reduce_add, ctx);
  VLOG(3) << "<<<< Done cpu_scatter_add_kernel <<<<<";
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_kernel(Tensor self, int dim, const Tensor& index,
                            Tensor src, const platform::DeviceContext& ctx) {
  VLOG(3) << "start scatter mul kernel";
  VLOG(3) << "src data scatter:" << *(src.data<tensor_t>());
  cpu_gather_scatter_functor<tensor_t, index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_mul_cpu", reduce_mul, ctx);
  VLOG(3) << "<<<< Done cpu_scatter_assign_kernel <<<<<";
}

namespace plat = paddle::platform;

Instantiate_Template_Funtion(cpu_gather_kernel)
    Instantiate_Template_Funtion(cpu_scatter_assign_kernel)
        Instantiate_Template_Funtion(cpu_scatter_add_kernel)
            Instantiate_Template_Funtion(cpu_scatter_mul_kernel)

}  // namespace operators
}  // namespace paddle
