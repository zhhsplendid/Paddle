// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/target.h"

namespace cinn {
namespace runtime {
class BackendAPI {
 public:
  BackendAPI() {};
  virtual ~BackendAPI() {};
  enum class MemcpyType : int{
    HostToHost=0,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
  };
  /*!
   * \brief Get BackendAPI by target.
   * \param target
   * \return The corresponding BackendAPI.
   */
  static BackendAPI* get_backend(const common::Target target);
  /*!
   * \brief Get BackendAPI by target name.
   * \param target_language
   * \return The corresponding BackendAPI.
   */
  static BackendAPI* get_backend(common::Target::Language target_language);
  /*!
   * \brief Set device by device_id
   * \param device_id
   */
  virtual void set_device(int device_id) =0;
  /*!
   * \brief Set active device by device_ids
   * \param device_ids
   */
  //virtual void set_active_devices(std::vector<int> device_ids) =0;
  /*!
   * \brief malloc memory in the idth device
   * \param numBytes
   * \param device_id
   * \return pointer to memory
   */
  virtual void* malloc(size_t numBytes) =0;
  /*!
   * \brief free memory in the idth device
   * \param data pointer to memory
   * \param device_id
   */
  virtual void free(void* data) =0;
  /*!
   * \brief  in the idth device
   * \param data pointer to memory
   * \param device_id
   */
  virtual void memset(void* data, int value, size_t numBytes) =0;
  virtual void memcpy(void* dest, const void* src, size_t numBytes, MemcpyType type) =0;
  /*!
   * \brief synchronize the idth device
   */
  virtual void device_sync() =0;

};
}  // namespace runtime
}  // namespace cinn