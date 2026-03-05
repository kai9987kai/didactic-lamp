#include "agents.h"
#include "evolution.h"
#include "metrics.h"
#include "types.h"
#include "world.h"

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int parse_int_arg(const std::string& flag, const std::string& value, int min_value) {
  try {
    return std::max(min_value, std::stoi(value));
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid integer for " + flag + ": " + value);
  }
}

float parse_float_arg(const std::string& flag, const std::string& value, float min_value) {
  try {
    return std::max(min_value, std::stof(value));
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid float for " + flag + ": " + value);
  }
}

void print_help() {
  std::cout
      << "Universe Simulation\n"
      << "Usage:\n"
      << "  universe_sim [agents] [ticks] [temperature]\n"
      << "  universe_sim [options]\n\n"
      << "Options:\n"
      << "  --agents N               Initial population (min 100)\n"
      << "  --ticks N                Simulation ticks (min 100)\n"
      << "  --temperature X          Softmax temperature (min 0.05)\n"
      << "  --snapshot N             Metrics interval (min 1)\n"
      << "  --seed N                 RNG seed\n"
      << "  --predator-ratio X       Predator population share\n"
      << "  --reproduction X         Reproduction energy threshold\n"
      << "  --speciation X           Species clustering distance\n"
      << "  --reproductive-distance X  Maximum mating genome distance\n"
      << "  --shock-interval N       Ticks between world shock windows\n"
      << "  --shock-duration N       Duration of each shock window\n"
      << "  --shock-strength X       Intensity of shock windows\n"
      << "  --help                   Show this message\n";
}

void apply_named_arg(const std::string& arg, const std::string& value, sim::Config& cfg) {
  if (arg == "--agents") {
    cfg.initial_agents = parse_int_arg(arg, value, 100);
  } else if (arg == "--ticks") {
    cfg.simulation_ticks = parse_int_arg(arg, value, 100);
  } else if (arg == "--temperature") {
    cfg.softmax_temperature = parse_float_arg(arg, value, 0.05f);
  } else if (arg == "--snapshot") {
    cfg.snapshot_interval = parse_int_arg(arg, value, 1);
  } else if (arg == "--seed") {
    cfg.seed = static_cast<uint64_t>(parse_int_arg(arg, value, 0));
  } else if (arg == "--predator-ratio") {
    cfg.predator_ratio = sim::clamp_value(parse_float_arg(arg, value, 0.0f), 0.0f, 0.95f);
  } else if (arg == "--reproduction") {
    cfg.reproduction_threshold = parse_float_arg(arg, value, 1.0f);
  } else if (arg == "--speciation") {
    cfg.speciation_threshold = parse_float_arg(arg, value, 0.05f);
  } else if (arg == "--reproductive-distance") {
    cfg.reproductive_distance = parse_float_arg(arg, value, 0.01f);
  } else if (arg == "--shock-interval") {
    cfg.shock_interval = parse_int_arg(arg, value, 1);
  } else if (arg == "--shock-duration") {
    cfg.shock_duration = parse_int_arg(arg, value, 1);
  } else if (arg == "--shock-strength") {
    cfg.shock_strength = parse_float_arg(arg, value, 0.0f);
  } else {
    throw std::runtime_error("Unknown option: " + arg);
  }
}

void parse_args(int argc, char** argv, sim::Config& cfg) {
  std::vector<std::string> positional;
  positional.reserve(3);

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_help();
      std::exit(0);
    }

    if (arg.rfind("--", 0) == 0) {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + arg);
      }
      apply_named_arg(arg, argv[++i], cfg);
      continue;
    }

    positional.push_back(arg);
  }

  if (!positional.empty()) cfg.initial_agents = parse_int_arg("agents", positional[0], 100);
  if (positional.size() > 1) cfg.simulation_ticks = parse_int_arg("ticks", positional[1], 100);
  if (positional.size() > 2) cfg.softmax_temperature = parse_float_arg("temperature", positional[2], 0.05f);
  if (positional.size() > 3) {
    throw std::runtime_error("Too many positional arguments. Use --help for the supported CLI.");
  }

  cfg.snapshot_interval = std::max(1, cfg.snapshot_interval);
  cfg.shock_duration = std::min(cfg.shock_duration, cfg.shock_interval);
}

}  // namespace

int main(int argc, char** argv) {
  sim::Config cfg;

  try {
    parse_args(argc, argv, cfg);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n\n";
    print_help();
    return 1;
  }

  std::cout << "=== Universe Simulation v4.0 (Niche + Shock Ecology) ===\n";
  std::cout << "Initial Agents: " << cfg.initial_agents
            << "  Ticks: " << cfg.simulation_ticks
            << "  Temp: " << cfg.softmax_temperature
            << "  Snapshot: " << cfg.snapshot_interval << "\n";
  std::cout << "Predator ratio: " << cfg.predator_ratio
            << "  Reproduction: " << cfg.reproduction_threshold
            << "  Mating distance: " << cfg.reproductive_distance << "\n";
  std::cout << "Shock cycle: " << cfg.shock_interval
            << "/" << cfg.shock_duration
            << "  Shock strength: " << cfg.shock_strength
            << "  Genome size: " << sim::kGenomeSize << "\n";
  std::cout << "Neural arch: " << sim::kInputFeatures << " -> "
            << sim::kHiddenNeurons << " -> " << sim::kOutputNodes
            << " (4 RNN memory outputs)\n\n";

  std::mt19937_64 rng(cfg.seed);
  sim::WorldFields world = sim::build_world(cfg);

  std::vector<sim::Agent> population;
  sim::init_population(population, cfg, rng);

  sim::SpeciesTracker species_tracker;
  std::vector<sim::Metrics> all_metrics;
  all_metrics.reserve(static_cast<size_t>(cfg.simulation_ticks / cfg.snapshot_interval + 1));

  float spec_threshold = cfg.speciation_threshold;
  species_tracker.classify(population, spec_threshold, 0);

  int last_snapshot_tick = -1;
  int births_since_snapshot = 0;
  int deaths_since_snapshot = 0;

  for (int tick = 0; tick <= cfg.simulation_ticks; ++tick) {
    if (population.empty()) {
      std::cout << "\n[EXTINCTION] All agents died at tick " << tick << ".\n";
      break;
    }

    sim::update_climate_and_resources(world, cfg, tick);
    sim::step_agents_movement(population, cfg, world, tick, rng);

    births_since_snapshot += sim::resolve_mating(population, cfg, rng, tick);
    deaths_since_snapshot += sim::cull_dead_agents(population);

    if (tick % cfg.snapshot_interval == 0) {
      species_tracker.classify(population, spec_threshold, tick);

      sim::Metrics m = sim::compute_metrics(population, world, cfg, tick,
                                            births_since_snapshot, deaths_since_snapshot);
      m.extinction_events = species_tracker.count_extinctions_since(last_snapshot_tick, tick);
      m.speciation_events = species_tracker.count_speciations_since(last_snapshot_tick, tick);
      all_metrics.push_back(m);

      std::cout << "tick=" << std::setw(5) << tick
                << " pop=" << population.size() << " (+" << births_since_snapshot
                << "/-" << deaths_since_snapshot << ")"
                << std::fixed << std::setprecision(4)
                << " fit=" << m.mean_fitness
                << " niche=" << m.mean_habitat_match
                << " res=" << m.mean_resources
                << " tox=" << m.mean_toxicity
                << " | species=" << m.species_count
                << " H=" << m.diversity_shannon
                << " event=" << sim::world_event_name(m.active_event)
                << ":" << m.event_intensity
                << "\n";

      births_since_snapshot = 0;
      deaths_since_snapshot = 0;
      last_snapshot_tick = tick;

      spec_threshold *= 0.9925f;
      spec_threshold = std::max(spec_threshold, 0.24f);
    }
  }

  std::ofstream out("simulation_summary.json", std::ios::binary);
  out << sim::summary_json(cfg, all_metrics, species_tracker.records);
  out.close();

  std::cout << "\nWrote simulation_summary.json ("
            << species_tracker.records.size() << " total species tracked)\n";

  return 0;
}
