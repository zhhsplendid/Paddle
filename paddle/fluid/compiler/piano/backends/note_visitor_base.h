// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

namespace paddle {
namespace piano {

template <typename InstructionPtr>
class NoteVisitorBase {
 public:
  virtual ~NoteVisitorBase() {}

  // Scalar op
  virtual void VisitConstant(InstructionPtr) = 0;

  // ops can be replaced by library
  virtual void VisitBatchNormGrad(InstructionPtr) = 0;
  virtual void VisitBatchNormInference(InstructionPtr) = 0;
  virtual void VisitBatchNormTraining(InstructionPtr) = 0;
  virtual void VisitConvolution(InstructionPtr) = 0;
  virtual void VisitDot(InstructionPtr) = 0;

  // Unary
  virtual void VisitBroadcast(InstructionPtr) = 0;
  virtual void VisitCast(InstructionPtr) = 0;
  virtual void VisitCopy(InstructionPtr) = 0;
  virtual void VisitExp(InstructionPtr) = 0;
  virtual void VisitLog(InstructionPtr) = 0;
  virtual void VisitNegative(InstructionPtr) = 0;
  virtual void VisitNot(InstructionPtr) = 0;
  virtual void VisitReshape(InstructionPtr) = 0;
  virtual void VisitReverse(InstructionPtr) = 0;
  virtual void VisitRsqrt(InstructionPtr) = 0;
  virtual void VisitSlice(InstructionPtr) = 0;
  virtual void VisitSqrt(InstructionPtr) = 0;
  virtual void VisitTranspose(InstructionPtr) = 0;

  // Binary
  virtual void VisitAdd(InstructionPtr) = 0;
  virtual void VisitAnd(InstructionPtr) = 0;
  virtual void VisitCompare(InstructionPtr) = 0;
  virtual void VisitDivide(InstructionPtr) = 0;
  virtual void VisitMaximum(InstructionPtr) = 0;
  virtual void VisitMinimum(InstructionPtr) = 0;
  virtual void VisitMultiply(InstructionPtr) = 0;
  virtual void VisitOr(InstructionPtr) = 0;
  virtual void VisitSubtract(InstructionPtr) = 0;
  virtual void VisitXor(InstructionPtr) = 0;

  // other
  virtual void VisitSelect(InstructionPtr) = 0;
  virtual void VisitConcatenate(InstructionPtr) = 0;
  virtual void VisitReduce(InstructionPtr) = 0;
  virtual void VisitRng(InstructionPtr) = 0;
  virtual void VisitSort(InstructionPtr) = 0;
  virtual void VisitTuple(InstructionPtr) = 0;
};

}  // namespace piano
}  // namespace paddle
