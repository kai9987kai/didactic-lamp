#include "agents.h"
#include "cli.h"
#include "evolution.h"
#include "metrics.h"
#include "world.h"
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  try {
    const auto options = sim::parse_args(argc, argv);
    if (options.help) { sim::print_help(); return 0; }
    const auto& cfg = options.config;
    std::ofstream out(options.output, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output: " + options.output);
    std::cout << "Didactic Lamp v5 | seed=" << cfg.seed << " | agents=" << cfg.initial_agents
              << " | ticks=" << cfg.simulation_ticks << " | grid=" << cfg.width << "x" << cfg.height << "\n";
    std::mt19937_64 rng(cfg.seed);
    auto world = sim::build_world(cfg);
    std::vector<sim::Agent> population;
    sim::init_population(population, cfg, rng);
    sim::rebuild_occupancy(population, world, cfg);
    sim::SpeciesTracker tracker;
    tracker.classify(population, cfg.speciation_threshold, 0);
    std::vector<sim::Metrics> metrics;
    sim::RunResult run;
    int births = 0, deaths = 0, last_snapshot = -1;
    auto snapshot = [&](int tick) {
      auto m = sim::compute_metrics(population, world, cfg, tick, births, deaths);
      m.extinction_events = tracker.count_extinctions_since(last_snapshot, tick);
      m.speciation_events = tracker.count_speciations_since(last_snapshot, tick);
      metrics.push_back(m);
      std::cout << "tick=" << std::setw(6) << tick << " pop=" << m.population
                << " (+" << births << "/-" << deaths << ") species=" << m.species_count
                << std::fixed << std::setprecision(3) << " effective=" << m.effective_species
                << " resources=" << m.mean_resources << "\n";
      births = deaths = 0;
      last_snapshot = tick;
    };
    snapshot(0);
    for (int tick = 1; tick <= cfg.simulation_ticks; ++tick) {
      sim::rebuild_occupancy(population, world, cfg);
      sim::update_climate_and_resources(world, cfg, tick);
      sim::step_agents_movement(population, cfg, world, tick, rng);
      const int born = sim::resolve_mating(population, cfg, rng, tick);
      const int died = sim::cull_dead_agents(population);
      sim::rebuild_occupancy(population, world, cfg);
      births += born;
      deaths += died;
      run.total_births += static_cast<uint64_t>(born);
      run.total_deaths += static_cast<uint64_t>(died);
      run.ticks_completed = tick;
      run.final_population = static_cast<int>(population.size());
      run.population_time_integral += population.size();
      const bool extinct = population.empty();
      const bool final = extinct || tick == cfg.simulation_ticks;
      // Classification is a model step; observing the model must not alter it.
      if (tick % cfg.classification_interval == 0 || final)
        tracker.classify(population, cfg.speciation_threshold, tick);
      if (extinct) run.extinction_tick = tick;
      if (tick % cfg.snapshot_interval == 0 || final) snapshot(tick);
      if (extinct) break;
    }
    out << sim::summary_json(cfg, metrics, tracker.records, run);
    out.flush();
    if (!out) throw std::runtime_error("Failed to write output: " + options.output);
    out.close();
    if (!out) throw std::runtime_error("Failed to close output: " + options.output);
    std::cout << "Wrote " << options.output << " | "
              << (run.extinction_tick >= 0 ? "extinct" : "completed")
              << " at tick " << run.ticks_completed << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\nUse --help for valid options.\n";
    return 1;
  }
}
