#include "../src/metrics.h"
#include "../src/world.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
bool close(float a, float b) { return std::abs(a - b) < 1e-5f; }

int main() {
  try {
    sim::Config cfg;
    cfg.width = cfg.height = 4;
    auto world = sim::build_world(cfg);
    std::fill(world.resources.begin(), world.resources.end(), 0.5f);
    std::fill(world.toxicity.begin(), world.toxicity.end(), 0.2f);
    std::fill(world.pheromone.begin(), world.pheromone.end(), 1.0f);
    std::vector<sim::Agent> agents(3);
    agents[0].fitness = -2;
    agents[1].fitness = -4;
    agents[1].species_id = 1;
    agents[2].alive = false;
    agents[2].fitness = 100;
    auto m = sim::compute_metrics(agents, world, cfg, 20, 2, 1);
    check(m.population == 2 && close(m.mean_fitness, -3) && close(m.max_fitness, -2),
          "dead agents must not affect the denominator or maximum");
    check(close(m.effective_species, 2) && close(m.inverse_simpson, 2) && close(m.species_evenness, 1),
          "equal abundance of two species must have two effective species");
    agents[1].species_id = 0;
    m = sim::compute_metrics(agents, world, cfg, 21, 0, 0);
    check(close(m.effective_species, 1) && close(m.inverse_simpson, 1) && close(m.species_evenness, 1),
          "singleton diversity convention must be explicit");
    agents.clear();
    m = sim::compute_metrics(agents, world, cfg, 22, 0, 2);
    check(m.population == 0 && m.species_count == 0 && m.effective_species == 0 && m.inverse_simpson == 0,
          "empty populations must have zero diversity");
    check(close(m.mean_resources, 0.5f) && close(m.mean_toxicity, 0.2f) && close(m.total_pheromone, 16),
          "extinction must not erase world measurements");
    cfg.shock_strength = 0;
    check(!sim::current_world_event(cfg, 180).active, "zero shock strength disables event labels");
    std::cout << "Metrics regression tests passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
