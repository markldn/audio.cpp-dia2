#include "engine/community_models/dia2/loader.h"
#include <iostream>
int main(int argc, char **argv) {
  try {
    if (argc < 3)
      throw std::runtime_error("Usage: dia2_parity_probe PACKAGE OUTPUT "
                               "[WEIGHTS] [BACKEND] [DEVICE] [THREADS]");
    engine::runtime::SessionOptions o;
    o.backend.threads = argc > 6 ? std::stoi(argv[6]) : 6;
    if (argc > 4)
      o.backend.type =
          (std::string(argv[4]) == "hip" ? engine::core::BackendType::Hip
                                         : engine::core::BackendType::Cpu);
    if (argc > 5)
      o.backend.device = std::stoi(argv[5]);
    engine::community_models::dia2::run_parity_probe(
        argv[1],
        argc > 3 ? argv[3] : (std::filesystem::path(argv[1]) / "dia2-f16.gguf"),
        o, argv[2]);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
