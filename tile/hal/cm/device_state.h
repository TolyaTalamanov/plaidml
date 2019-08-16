// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "base/context/context.h"
#include "tile/hal/cm/cm.pb.h"
#include "tile/hal/cm/runtime.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

// DeviceState represents the state of a device, including all cm objects needed
// to control the device.
class DeviceState {
 public:
  struct QueueStruct {
    CmQueue* pCmQueue_;
  };
  void Flush() const;

  DeviceState(const context::Context& ctx, CmDevice* pCmDev, proto::DeviceInfo dinfo);

  ~DeviceState();

  void Initialize();

  const proto::DeviceInfo info() const { return info_; }
  CmDevice* cmdev() { return pCmDev_; }
  const context::Clock& clock() const { return clock_; }
  const context::proto::ActivityID& id() const { return id_; }

  void FlushCommandQueue();

  std::unique_ptr<QueueStruct> cm_queue_;

 private:
  CmDevice* pCmDev_;
  const proto::DeviceInfo info_;
  const context::Clock clock_;
  const context::proto::ActivityID id_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
