#include "agents.h"
#include "evolution.h"
#include "metrics.h"
#include "types.h"
#include "world.h"

#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  sim::Config cfg;
  if (argc > 1) cfg.initial_agents = std::max(100, std::stoi(argv[1]));
  if (argc > 2) cfg.simulation_ticks = std::max(100, std::stoi(argv[2]));
  if (argc > 3) cfg.softmax_temperature = std::max(0.05f, std::stof(argv[3]));

  std::cout << "=== Universe Simulation v3.0 (Phase 2 Continuous) ===\n";
  std::cout << "Initial Agents: " << cfg.initial_agents
            << "  Ticks: " << cfg.simulation_ticks
            << "  Temp: " << cfg.softmax_temperature << "\n";
  std::cout << "Predator ratio: " << cfg.predator_ratio
            << "  Hunt prob: " << cfg.hunt_success_prob
            << "  Genome size: " << sim::kGenomeSize << "\n";
  std::cout << "Neural arch: " << sim::kInputFeatures << " -> "
            << sim::kHiddenNeurons << " -> " << sim::kOutputNodes 
            << " (4 RNN Memory States)\n\n";

  std::mt19937_64 rng(cfg.seed);
  sim::WorldFields world = sim::build_world(cfg);

  std::vector<sim::Agent> population;
  sim::init_population(population, cfg, rng);

  sim::SpeciesTracker species_tracker;
  std::vector<sim::Metrics> all_metrics;
  all_metrics.reserve(static_cast<size_t>(cfg.simulation_ticks / cfg.snapshot_interval + 1));

  float spec_threshold = cfg.speciation_threshold;
  
  int last_snapshot_tick = 0;
  int births_since_snapshot = 0;
  int deaths_since_snapshot = 0;

  for (int tick = 0; tick <= cfg.simulation_ticks; ++tick) {
    if (population.empty()) {
       std::cout << "\n[EXTINCTION] All agents died at tick " << tick << ".\n";
       break;
    }

    // ── Continuous Environment & Agent Execution ──
    sim::update_climate_and_resources(world, cfg, tick);
    sim::step_agents_movement(population, cfg, world, tick, rng);

    // ── Asynchronous Evolution (Breeding & Dying) ──
    births_since_snapshot += sim::resolve_mating(population, cfg, rng, tick);
    deaths_since_snapshot += sim::cull_dead_agents(population);

    // ── Metrics Snapshot & Speciation ──
    if (tick % cfg.snapshot_interval == 0) {
      // Re-classify species periodically
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
                << " toxR=" << m.mean_tox_res
                << " spd=" << m.mean_speed
                << " sz=" << m.mean_size
                << " | species=" << m.species_count
                << " H=" << m.diversity_shannon
                << "\n";

      births_since_snapshot = 0;
      deaths_since_snapshot = 0;
      last_snapshot_tick = tick;

      // Tighten speciation threshold over long timescales
      spec_threshold *= 0.99f;
      spec_threshold = std::max(spec_threshold, 0.3f);
    }
  }

  std::ofstream out("simulation_summary.json", std::ios::binary);
  out << sim::summary_json(cfg, all_metrics, species_tracker.records);
  out.close();
  
  std::cout << "\nWrote simulation_summary.json ("
            << species_tracker.records.size() << " total species tracked)\n";

  return 0;
}
