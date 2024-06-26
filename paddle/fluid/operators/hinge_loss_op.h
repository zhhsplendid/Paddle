/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext, typename AttrType = T>
class HingeLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* pred = context.Input<phi::DenseTensor>("Logits");
    auto* label = context.Input<phi::DenseTensor>("Labels");
    auto* loss = context.Output<phi::DenseTensor>("Loss");
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto x = phi::EigenVector<T>::Flatten(*pred);
    auto y = phi::EigenVector<T>::Flatten(*label);
    loss->mutable_data<T>(context.GetPlace());
    auto l = phi::EigenVector<T>::Flatten(*loss);
    phi::funcs::EigenHingeLoss<std::decay_t<decltype(place)>, T>::Eval(
        place, l, x, y);
  }
};

template <typename T, typename DeviceContext, typename AttrType = T>
class HingeLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* pred = context.Input<phi::DenseTensor>("Logits");
    auto* label = context.Input<phi::DenseTensor>("Labels");
    auto* dloss =
        context.Input<phi::DenseTensor>(framework::GradVarName("Loss"));
    auto* dpred =
        context.Output<phi::DenseTensor>(framework::GradVarName("Logits"));
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto x = phi::EigenVector<T>::Flatten(*pred);
    auto y = phi::EigenVector<T>::Flatten(*label);
    auto dl = phi::EigenVector<T>::Flatten(*dloss);

    if (dpred) {
      dpred->mutable_data<T>(context.GetPlace());
      auto dx = phi::EigenVector<T>::Flatten(*dpred);
      phi::funcs::EigenHingeLossGrad<std::decay_t<decltype(place)>, T>::Eval(
          place, dx, dl, x, y);
    }
  }
};

}  // namespace operators
}  // namespace paddle
