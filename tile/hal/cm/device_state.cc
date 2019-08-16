// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/device_state.h"

#include <string>
#include <utility>
#include <vector>

#include "base/util/error.h"
#include "tile/hal/cm/err.h"
#include "tile/hal/cm/runtime.h"
#include "tile/hal/util/selector.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {
namespace {

DeviceState::QueueStruct MakeQueue(CmDevice* pCmDev) {
  DeviceState::QueueStruct result;

  pCmDev->InitPrintBuffer();
  CmQueue* pCmQueue = NULL;
  pCmDev->CreateQueue(pCmQueue);

  result.pCmQueue_ = pCmQueue;
  return result;
}

}  // namespace

void DeviceState::Flush() const { cm_result_check(pCmDev_->FlushPrintBuffer()); }

DeviceState::DeviceState(const context::Context& ctx, CmDevice* pCmDev, proto::DeviceInfo dinfo)
    : cm_queue_{std::unique_ptr<QueueStruct>()},
      pCmDev_{pCmDev},
      info_{std::move(dinfo)},
      clock_{},
      id_{ctx.activity_id()} {}

DeviceState::~DeviceState() { cm_result_check(::DestroyCmDevice(pCmDev_)); }
void DeviceState::Initialize() { cm_queue_ = std::make_unique<QueueStruct>(MakeQueue(pCmDev_)); }

void DeviceState::FlushCommandQueue() { Flush(); }

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
