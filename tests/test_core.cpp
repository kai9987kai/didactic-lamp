#include "../src/agents.h"
#include "../src/evolution.h"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

sim::Agent parent(sim::Gender gender, float energy = 12.0f) {
  sim::Agent a;
  a.pos = {2.0f, 2.0f};
  a.gender = gender;
  a.energy = energy;
  sim::decode_morphology(a);
  return a;
}

sim::Config small_config() {
  sim::Config cfg;
  cfg.width = 8;
  cfg.height = 8;
  cfg.max_agents = 100;
  cfg.reproductive_distance = 1.0f;
  cfg.shock_strength = 0.0f;
  return cfg;
}

sim::WorldFields flat_world(const sim::Config& cfg) {
  auto world = sim::build_world(cfg);
  std::fill(world.height.begin(), world.height.end(), 0.3f);
  std::fill(world.temperature.begin(), world.temperature.end(), 0.55f);
  std::fill(world.moisture.begin(), world.moisture.end(), 0.35f);
  std::fill(world.resources.begin(), world.resources.end(), 0.0f);
  std::fill(world.toxicity.begin(), world.toxicity.end(), 0.0f);
  std::fill(world.biome.begin(), world.biome.end(), static_cast<uint8_t>(sim::Biome::Grassland));
  return world;
}

void test_population_cap() {
  auto cfg = small_config();
  cfg.max_agents = 5;
  std::vector<sim::Agent> agents{
      parent(sim::Gender::Female), parent(sim::Gender::Male),
      parent(sim::Gender::Female), parent(sim::Gender::Male)};
  std::mt19937_64 rng(7);
  for (auto& a : agents) a.species_id = 37;
  const int births = sim::resolve_mating(agents, cfg, rng, 100);
  check(births == 1 && agents.size() == 5, "mating must enforce the cap after every birth");
  check(agents.back().species_id == 37, "newborns must inherit a provisional species until classification");
  check(sim::resolve_mating(agents, cfg, rng, 200) == 0, "full population must not reproduce");
}

void test_newborns_and_energy() {
  auto cfg = small_config();
  cfg.reproduction_threshold = 1.0f;
  for (uint64_t seed = 0; seed < 32; ++seed) {
    std::vector<sim::Agent> agents{
        parent(sim::Gender::Female), parent(sim::Gender::Male), parent(sim::Gender::Male)};
    std::mt19937_64 rng(seed);
    check(sim::resolve_mating(agents, cfg, rng, 100) == 1,
          "newborns must not enter their birth tick's mating loop");
  }
  std::vector<sim::Agent> agents{parent(sim::Gender::Female, 4.0f), parent(sim::Gender::Male, 4.0f)};
  std::mt19937_64 rng(7);
  check(sim::resolve_mating(agents, cfg, rng, 100) == 0,
        "parents must afford the energy contribution even below the configured threshold");
}

void test_cooldown() {
  auto cfg = small_config();
  std::vector<sim::Agent> agents{parent(sim::Gender::Female), parent(sim::Gender::Male)};
  for (auto& a : agents) a.last_mate_tick = 70;
  std::mt19937_64 rng(7);
  check(sim::resolve_mating(agents, cfg, rng, 99) == 0, "cooldown must block mating for 29 ticks");
  check(sim::resolve_mating(agents, cfg, rng, 100) == 1, "both parents become eligible at 30 ticks");
}

void test_genome_and_species() {
  auto a = parent(sim::Gender::Female);
  auto b = parent(sim::Gender::Male);
  b.genome.back() = 1.0f;
  check(sim::genetic_distance(a, b) > 0.05f, "genetic distance must include niche genes at genome tail");
  sim::SpeciesTracker tracker;
  std::vector<sim::Agent> agents{a, b};
  tracker.classify(agents, 0.01f, 0);
  check(agents[0].species_id != agents[1].species_id, "species signatures must include the whole genome");
  b = a;
  b.type = sim::AgentType::Predator;
  agents = {a, b};
  tracker = {};
  tracker.classify(agents, 1.0f, 0);
  check(agents[0].species_id != agents[1].species_id, "herbivores and predators must occupy different clusters");
}

void test_hunting_boundaries() {
  auto cfg = small_config();
  for (uint64_t seed = 0; seed < 32; ++seed) {
    auto world = flat_world(cfg);
    auto hunter = parent(sim::Gender::Female);
    hunter.type = sim::AgentType::Predator;
    hunter.pos = {0.0f, 0.0f};
    auto prey = parent(sim::Gender::Male);
    prey.pos = hunter.pos;
    std::vector<sim::Agent> agents{hunter, prey};
    std::mt19937_64 expected_rng(seed);
    std::uniform_real_distribution<float> roll(0.0f, 1.0f);
    const bool expected_kill = roll(expected_rng) < 0.17f;
    std::mt19937_64 rng(seed);
    sim::resolve_hunting(agents, cfg, world, rng);
    check(!agents[1].alive == expected_kill, "a boundary prey must receive only one hunting attempt");
    check(rng == expected_rng, "out-of-bounds offsets must not consume extra hunt rolls");
  }
  auto world = flat_world(cfg);
  auto hunter = parent(sim::Gender::Female, 0.01f);
  hunter.type = sim::AgentType::Predator;
  std::vector<sim::Agent> agents{hunter};
  std::mt19937_64 rng(7);
  sim::resolve_hunting(agents, cfg, world, rng);
  check(!agents[0].alive, "a predator exhausting energy in a failed hunt must die in that tick");
}

void test_zero_move_metabolism() {
  auto cfg = small_config();
  uint64_t seed = 0;
  for (;; ++seed) {
    std::mt19937_64 probe(seed);
    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(probe) < 0.5f) break;
  }
  auto world = flat_world(cfg);
  auto a = parent(sim::Gender::Female);
  a.speed_mod = 0.5f;
  a.metabolic_rate = 0.1f;
  std::vector<sim::Agent> agents{a};
  std::mt19937_64 rng(seed);
  sim::step_agents_movement(agents, cfg, world, 1, rng);
  check(agents[0].energy < a.energy, "a zero-move tick still incurs basal metabolism");
  check(agents[0].pos.x == a.pos.x && agents[0].pos.y == a.pos.y,
        "the metabolism regression must actually exercise a zero-move tick");
}

void test_live_occupancy() {
  auto cfg = small_config();
  auto world = flat_world(cfg);
  auto first = parent(sim::Gender::Female);
  first.body_size = 1.5f;
  auto second = parent(sim::Gender::Male);
  second.body_size = 0.75f;
  auto dead = first;
  dead.alive = false;
  std::vector<sim::Agent> agents{first, second, dead};
  std::fill(world.occupancy.begin(), world.occupancy.end(), 99.0f);
  sim::rebuild_occupancy(agents, world, cfg);
  const size_t occupied_cell = sim::idx_2d(2, 2, cfg);
  for (size_t i = 0; i < world.occupancy.size(); ++i) {
    check(world.occupancy[i] == (i == occupied_cell ? 2.25f : 0.0f),
          "occupancy must contain current living body mass exactly once");
  }
  const auto previous = world.occupancy;
  sim::update_climate_and_resources(world, cfg, 1);
  check(world.occupancy == previous, "world updates must not decay or mutate live occupancy");
  agents[1].alive = false;
  agents[0].pos = {4.0f, 4.0f};
  sim::rebuild_occupancy(agents, world, cfg);
  check(world.occupancy[occupied_cell] == 0.0f && world.occupancy[sim::idx_2d(4, 4, cfg)] == 1.5f,
        "occupancy must reflect movement and deaths without a historical trail");
}

void test_resource_recovery() {
  auto cfg = small_config();
  auto world = flat_world(cfg);
  cfg.resource_recovery = 0.0f;
  sim::update_climate_and_resources(world, cfg, 1);
  check(std::all_of(world.resources.begin(), world.resources.end(), [](float r) { return r == 0.0f; }),
        "with recovery disabled and no event, depleted resources stay depleted");
  cfg.resource_recovery = 0.02f;
  sim::update_climate_and_resources(world, cfg, 2);
  check(std::all_of(world.resources.begin(), world.resources.end(), [](float r) { return r > 0.0f && r <= 0.8f; }),
        "positive recovery restores empty habitat within its carrying capacity");
  world = flat_world(cfg);
  std::fill(world.resources.begin(), world.resources.end(), 1.2f);
  auto control = world;
  cfg.resource_recovery = 0.0f;
  sim::update_climate_and_resources(control, cfg, 1);
  cfg.resource_recovery = 1.0f;
  sim::update_climate_and_resources(world, cfg, 1);
  check(world.resources == control.resources, "recovery must not subtract resources above carrying capacity");
}

}  // namespace

int main() {
  try {
    test_population_cap();
    test_newborns_and_energy();
    test_cooldown();
    test_genome_and_species();
    test_hunting_boundaries();
    test_zero_move_metabolism();
    test_live_occupancy();
    test_resource_recovery();
    std::cout << "Core regression tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Core regression failure: " << error.what() << '\n';
    return 1;
  }
}
