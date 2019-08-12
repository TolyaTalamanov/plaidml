// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/cm_library.h"

#include <utility>

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

cmLibrary* cmLibrary::Downcast(hal::Library* cmlibrary, std::shared_ptr<cmDeviceState> device_state) {
  cmLibrary* exe = dynamic_cast<cmLibrary*>(cmlibrary);
  if (!exe || exe->device_state() != device_state) {
    throw error::InvalidArgument{"Incompatible library for Tile device"};
  }
  return exe;
}

cmLibrary::cmLibrary(std::shared_ptr<cmDeviceState> device_state, const std::map<std::string, CmProgram*>& program_map,
                     const std::map<std::string, std::shared_ptr<Emit>>& emit_map,
                     const std::vector<lang::KernelInfo>& kernel_info,
                     std::vector<context::proto::ActivityID> kernel_ids)
    : device_state_{device_state},
      program_map_{std::move(program_map)},
      emit_map_{std::move(emit_map)},
      kernel_info_{kernel_info},
      kernel_ids_{std::move(kernel_ids)} {}

std::map<std::string, std::string> cmLibrary::Serialize() {
  std::map<std::string, std::string> result;
  return result;
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
