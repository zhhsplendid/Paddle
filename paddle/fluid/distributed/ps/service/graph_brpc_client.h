// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <ThreadPool.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ThreadPool.h"
#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_server.h"
#include "paddle/fluid/distributed/ps/service/ps_client.h"
#include "paddle/fluid/distributed/ps/table/table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace distributed {

class GraphPsService_Stub : public PsService_Stub {
 public:
  GraphPsService_Stub(::google::protobuf::RpcChannel* channel,
                      ::google::protobuf::RpcChannel* local_channel = NULL,
                      GraphBrpcService* service = NULL,
                      int thread_num = 1)
      : PsService_Stub(channel) {
    this->local_channel = local_channel;
    this->graph_service = service;
    task_pool.reset(new ::ThreadPool(thread_num));
  }
  virtual ~GraphPsService_Stub() {}

  GraphPsService_Stub(const GraphPsService_Stub&);
  GraphPsService_Stub& operator=(const GraphPsService_Stub&);

  // implements PsService ------------------------------------------
  GraphBrpcService* graph_service;
  std::shared_ptr<::ThreadPool> task_pool;
  ::google::protobuf::RpcChannel* local_channel;
  void service(::google::protobuf::RpcController* controller,
               const ::paddle::distributed::PsRequestMessage* request,
               ::paddle::distributed::PsResponseMessage* response,
               ::google::protobuf::Closure* done);
};
class GraphBrpcClient : public BrpcPsClient {
 public:
  GraphBrpcClient() {}
  virtual ~GraphBrpcClient() {}
  // given a batch of nodes, sample graph_neighbors for each of them
  virtual std::future<int32_t> batch_sample_neighbors(
      uint32_t table_id,
      int idx,
      std::vector<int64_t> node_ids,
      int sample_size,
      std::vector<std::vector<int64_t>>& const res,
      std::vector<std::vector<float>>& const res_weight,
      bool need_weight,
      int server_index = -1);

  virtual std::future<int32_t> pull_graph_list(
      uint32_t table_id,
      int type_id,
      int idx,
      int server_index,
      int start,
      int size,
      int step,
      std::vector<FeatureNode>& const res);

  virtual std::future<int32_t> random_sample_nodes(
      uint32_t table_id,
      int type_id,
      int idx,
      int server_index,
      int sample_size,
      std::vector<int64_t>& const ids);
  virtual std::future<int32_t> get_node_feat(
      const uint32_t& table_id,
      int idx,
      const std::vector<int64_t>& node_ids,
      const std::vector<std::string>& feature_names,
      std::vector<std::vector<std::string>>& const res);
  virtual std::future<int32_t> set_node_feat(
      const uint32_t& table_id,
      int idx,
      const std::vector<int64_t>& node_ids,
      const std::vector<std::string>& feature_names,
      const std::vector<std::vector<std::string>>& features);

  virtual std::future<int32_t> clear_nodes(uint32_t table_id,
                                           int type_id,
                                           int idx);
  virtual std::future<int32_t> add_graph_node(
      uint32_t table_id,
      int idx,
      std::vector<int64_t>& const node_id_list,
      std::vector<bool>& const is_weighted_list);
  virtual std::future<int32_t> remove_graph_node(
      uint32_t table_id, int idx_, std::vector<int64_t>& const node_id_list);
  virtual int32_t Initialize();
  int get_shard_num() { return shard_num; }
  void set_shard_num(int shard_num) { this->shard_num = shard_num; }
  int get_server_index_by_id(int64_t id);
  void set_local_channel(int index) {
    this->local_channel = GetCmdChannel(index);
  }
  void set_local_graph_service(GraphBrpcService* graph_service) {
    this->graph_service = graph_service;
  }
  GraphPsService_Stub getServiceStub(::google::protobuf::RpcChannel* channel,
                                     int thread_num = 1) {
    return GraphPsService_Stub(
        channel, local_channel, graph_service, thread_num);
  }

 private:
  int shard_num;
  size_t server_size;
  ::google::protobuf::RpcChannel* local_channel;
  GraphBrpcService* graph_service;
};

}  // namespace distributed
}  // namespace paddle
