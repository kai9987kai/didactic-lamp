#pragma once
#include "types.h"
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sim {
struct Options {
  Config config;
  std::string output{"simulation_summary.json"};
  bool help{false};
};

inline uint64_t parse_unsigned(const std::string& flag, const std::string& text) {
  if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c); }))
    throw std::runtime_error(flag + " requires an unsigned integer: " + text);
  try {
    size_t used = 0;
    const auto value = std::stoull(text, &used);
    if (used != text.size()) throw std::invalid_argument("trailing text");
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("Out-of-range integer for " + flag + ": " + text);
  }
}

inline int parse_integer(const std::string& flag, const std::string& text, int minimum, int maximum) {
  const auto value = parse_unsigned(flag, text);
  if (value < static_cast<uint64_t>(minimum) || value > static_cast<uint64_t>(maximum))
    throw std::runtime_error(flag + " must be in [" + std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
  return static_cast<int>(value);
}

inline float parse_real(const std::string& flag, const std::string& text, float minimum, float maximum) {
  float value;
  try {
    size_t used = 0;
    value = std::stof(text, &used);
    if (used != text.size()) throw std::invalid_argument("trailing text");
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid number for " + flag + ": " + text);
  }
  if (!std::isfinite(value) || value < minimum || value > maximum)
    throw std::runtime_error(flag + " must be finite and in [" + std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
  return value;
}

inline void print_help() {
  std::cout << "Didactic Lamp | Ecosystem research sandbox v5\n"
    "Usage: universe_sim [agents] [ticks] [temperature]\n"
    "       universe_sim [options]\n\n"
    "  --agents N                 Initial population [1, 100000]\n"
    "  --max-agents N             Hard population cap [1, 100000]\n"
    "  --ticks N                  Number of transitions [1, 10000000]\n"
    "  --width N --height N       Grid dimensions [2, 512]\n"
    "  --snapshot N               Observation interval; no effect on dynamics\n"
    "  --classification N         Species classification interval (default 25)\n"
    "  --seed N                   Unsigned 64-bit seed\n"
    "  --temperature X            Policy softmax temperature [0.05, 100]\n"
    "  --predator-ratio X          Initial predator fraction [0, 1]\n"
    "  --hunt-success X            Base hunt probability [0, 1] (default .12)\n"
    "  --reproduction X            Energy threshold [5, 1000]\n"
    "  --speciation X              Full-genome RMS clustering distance [0.01, 10]\n"
    "  --reproductive-distance X   Full-genome RMS mating distance [0.01, 10] (default .5)\n"
    "  --shock-interval N          Ticks between shocks (0 disables)\n"
    "  --shock-duration N          Shock duration (0 disables)\n"
    "  --shock-strength X          Shock strength [0, 1]; 0 disables\n"
    "  --resource-recovery X       Seed-bank recovery rate [0, 1] (default .002)\n"
    "  --output PATH              JSON destination (default simulation_summary.json)\n"
    "  --help                     Show this message\n\n"
    "Tick 0 is the untouched initial state. Runs include their final state,\n"
    "even when extinction occurs between observations. Units are abstract.\n";
}

inline Options parse_args(int argc, char** argv) {
  Options options;
  auto& cfg = options.config;
  std::vector<std::string> positional;
  bool named_dynamics = false;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--help" || flag == "-h") { options.help = true; return options; }
    if (flag.rfind("--", 0) != 0) { positional.push_back(flag); continue; }
    if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
    const std::string value = argv[++i];
    if (flag == "--output") {
      if (value.empty()) throw std::runtime_error("--output cannot be empty");
      options.output = value;
    } else {
      named_dynamics = true;
      if (flag == "--agents") cfg.initial_agents = parse_integer(flag, value, 1, 100000);
      else if (flag == "--max-agents") cfg.max_agents = parse_integer(flag, value, 1, 100000);
      else if (flag == "--ticks") cfg.simulation_ticks = parse_integer(flag, value, 1, 10000000);
      else if (flag == "--width") cfg.width = parse_integer(flag, value, 2, 512);
      else if (flag == "--height") cfg.height = parse_integer(flag, value, 2, 512);
      else if (flag == "--snapshot") cfg.snapshot_interval = parse_integer(flag, value, 1, 10000000);
      else if (flag == "--classification") cfg.classification_interval = parse_integer(flag, value, 1, 10000000);
      else if (flag == "--seed") cfg.seed = parse_unsigned(flag, value);
      else if (flag == "--temperature") cfg.softmax_temperature = parse_real(flag, value, 0.05f, 100.0f);
      else if (flag == "--predator-ratio") cfg.predator_ratio = parse_real(flag, value, 0.0f, 1.0f);
      else if (flag == "--hunt-success") cfg.hunt_success_prob = parse_real(flag, value, 0.0f, 1.0f);
      else if (flag == "--reproduction") cfg.reproduction_threshold = parse_real(flag, value, 5.0f, 1000.0f);
      else if (flag == "--speciation") cfg.speciation_threshold = parse_real(flag, value, 0.01f, 10.0f);
      else if (flag == "--reproductive-distance") cfg.reproductive_distance = parse_real(flag, value, 0.01f, 10.0f);
      else if (flag == "--shock-interval") cfg.shock_interval = parse_integer(flag, value, 0, 10000000);
      else if (flag == "--shock-duration") cfg.shock_duration = parse_integer(flag, value, 0, 10000000);
      else if (flag == "--shock-strength") cfg.shock_strength = parse_real(flag, value, 0.0f, 1.0f);
      else if (flag == "--resource-recovery") cfg.resource_recovery = parse_real(flag, value, 0.0f, 1.0f);
      else throw std::runtime_error("Unknown option: " + flag);
    }
  }
  if (positional.size() > 3) throw std::runtime_error("At most three positional arguments are supported");
  if (!positional.empty() && named_dynamics)
    throw std::runtime_error("Use either positional or named simulation options, not both");
  if (!positional.empty()) cfg.initial_agents = parse_integer("agents", positional[0], 1, 100000);
  if (positional.size() > 1) cfg.simulation_ticks = parse_integer("ticks", positional[1], 1, 10000000);
  if (positional.size() > 2) cfg.softmax_temperature = parse_real("temperature", positional[2], 0.05f, 100.0f);
  if (cfg.initial_agents > cfg.max_agents) throw std::runtime_error("--agents cannot exceed --max-agents");
  if (cfg.shock_interval > 0 && cfg.shock_duration > cfg.shock_interval)
    throw std::runtime_error("--shock-duration cannot exceed --shock-interval");
  return options;
}
} // namespace sim
