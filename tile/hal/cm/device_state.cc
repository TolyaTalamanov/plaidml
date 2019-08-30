// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/device_state.h"

#include <string>
#include <utility>
#include <vector>

#include "base/util/env.h"
#include "base/util/error.h"
#include "tile/hal/cm/err.h"
#include "tile/hal/cm/runtime.h"
#include "tile/hal/util/selector.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {
namespace {

DeviceState::QueueStruct* MakeQueue(CmDevice* pCmDev) {
  DeviceState::QueueStruct* result = new DeviceState::QueueStruct();

  pCmDev->InitPrintBuffer();
  CmQueue* pCmQueue = NULL;
  pCmDev->CreateQueue(pCmQueue);

  result->pCmQueue_ = pCmQueue;
  return result;
}

}  // namespace

void DeviceState::Flush() const { cm_result_check(pCmDev_->FlushPrintBuffer()); }

DeviceState::DeviceState(const context::Context& ctx, proto::DeviceInfo dinfo)
    : info_{std::move(dinfo)}, clock_{}, id_{ctx.activity_id()} {}

DeviceState::~DeviceState() {
  env::Set("USE_STRIPE", "");
  env::Set("PLAIDML_PROHIBIT_WINOGRAD", "");
  if (pCmDev_) cm_result_check(::DestroyCmDevice(pCmDev_));

  auto str = env::Get("PWD");
  std::string cmd = "rm -r " + str + "/cmkernels";
  auto check_err = system(cmd.c_str());

  if (check_err) {
    check_err = 0;
  }
}

void DeviceState::MakeDevice() {
  env::Set("LIBVA_DRIVER_NAME", "iHD");
  env::Set("LIBVA_DRIVERS_PATH", "/usr/lib/x86_64-linux-gnu/dri");

  UINT version = 0;
  cm_result_check(::CreateCmDevice(pCmDev_, version));
  if (version < CM_1_0) {
    throw std::runtime_error(std::string("The runtime API version is later than runtime DLL version "));
  }
  env::Set("USE_STRIPE", "1");
  env::Set("PLAIDML_PROHIBIT_WINOGRAD", "1");
  auto str = env::Get("PWD");
  std::string cmd = "mkdir " + str + "/cmkernels";
  auto check_err = system(cmd.c_str());

  if (check_err) {
    check_err = 0;
  }

  env::Set("PLAIDML_CM_CACHE", str + "/cmkernels");

  std::string prefix = str.substr(0, str.find("execroot"));

  env::Set("CM_ROOT", prefix + "external/cm_headers");

  cm_queue_ = MakeQueue(pCmDev_);
}

void DeviceState::FlushCommandQueue() { Flush(); }

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
