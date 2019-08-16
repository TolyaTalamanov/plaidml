// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/device.h"

#include <utility>

#include "base/util/compat.h"
#include "tile/hal/cm/compiler.h"
#include "tile/hal/cm/executor.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

Device::Device(const context::Context& ctx, CmDevice* pCmDev, proto::DeviceInfo dinfo)
    : device_state_{std::make_shared<DeviceState>(ctx, pCmDev, std::move(dinfo))},
      compiler_{std::make_unique<Compiler>(device_state_)},
      executor_{std::make_unique<Executor>(device_state_)} {}

void Device::Initialize(const hal::proto::HardwareSettings& settings) { device_state_->Initialize(); }

std::string Device::description() {  //
  return device_state()->info().vendor() + " " + device_state()->info().name() + " (CM)";
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
