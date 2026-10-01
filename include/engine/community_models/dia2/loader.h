#pragma once
#include "engine/framework/runtime/model.h"
namespace engine::community_models::dia2 {
std::shared_ptr<runtime::IVoiceModelLoader> make_dia2_loader();
}
namespace engine::community_models::dia2 {
void run_parity_probe(const std::filesystem::path &root,
                      const std::filesystem::path &weights,
                      const runtime::SessionOptions &options,
                      const std::filesystem::path &output);
}
