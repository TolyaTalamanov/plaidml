// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "base/context/context.h"
#include "tile/base/hal.h"
#include "tile/hal/cm/cm.pb.h"
#include "tile/hal/cm/cm_device.h"
#include "tile/hal/cm/cm_host_memory.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

// DeviceSet implements the hal::DeviceSet model as a single cm platform.
class cmDeviceSet final : public hal::DeviceSet {
 public:
  explicit cmDeviceSet(const context::Context& ctx);

  const std::vector<std::shared_ptr<hal::Device>>& devices() final;

  Memory* host_memory() final;

 private:
  std::vector<std::shared_ptr<hal::Device>> devices_;
  std::unique_ptr<Memory> host_memory_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
