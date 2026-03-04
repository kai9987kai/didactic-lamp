#pragma once
#include "types.h"
#include "world.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace sim {

// ── Neural Policy Network (2-layer) ──────────────────────────────────────────
// Genome layout:
//   [0 .. kInputFeatures*kHiddenNeurons-1]       = input→hidden weights
//   [kInputFeatures*kHiddenNeurons .. +kHiddenNeurons-1] = hidden biases
//   [... +kHiddenNeurons*kActions-1]              = hidden→output weights
//   [... +kActions-1]                             = output biases

inline std::array<float, kActions> policy_logits(const Agent& a, const Config& cfg,
                                                  const WorldFields& world, int x, int y,
                                                  float season_phase) {
  const size_t i = idx_2d(x, y, cfg);
  const float nx = a.pos.x / static_cast<float>(cfg.width - 1);
  const float ny = a.pos.y / static_cast<float>(cfg.height - 1);
  const float h = world.height[i];
  const float t = world.temperature[i];
  const float r = world.resources[i];
  const float density = mean3x3(world.occupancy, x, y, cfg);
  const float e = a.energy / 10.0f;
  const float pheromone = std::min(world.pheromone[i], 5.0f) / 5.0f;  // normalized
  const float moist = world.moisture[i];
  const float biome_f = static_cast<float>(world.biome[i]) / 5.0f;    // normalized 0-1

  std::array<float, kInputFeatures> feat{nx, ny, h, t, r, density, e, season_phase,
                                          pheromone, moist, biome_f};

  // Layer 1: input → hidden (tanh activation)
  const int w1_size = kInputFeatures * kHiddenNeurons;
  const int b1_offset = w1_size;
  const int w2_offset = b1_offset + kHiddenNeurons;
  const int b2_offset = w2_offset + kHiddenNeurons * kActions;

  std::array<float, kHiddenNeurons> hidden{};
  for (int hid = 0; hid < kHiddenNeurons; ++hid) {
    float s = a.genome[b1_offset + hid];  // bias
    for (int inp = 0; inp < kInputFeatures; ++inp) {
      s += a.genome[hid * kInputFeatures + inp] * feat[inp];
    }
    hidden[hid] = std::tanh(s);
  }

  // Layer 2: hidden → output
  std::array<float, kActions> logits{};
  for (int act = 0; act < kActions; ++act) {
    float s = a.genome[b2_offset + act];  // bias
    for (int hid = 0; hid < kHiddenNeurons; ++hid) {
      s += a.genome[w2_offset + act * kHiddenNeurons + hid] * hidden[hid];
    }
    logits[act] = s;
  }
  return logits;
}

inline int sample_action(const std::array<float, kActions>& logits, float temperature,
                          std::mt19937_64& rng) {
  const float inv_t = 1.0f / std::max(0.05f, temperature);
  float max_logit = *std::max_element(logits.begin(), logits.end());

  std::array<float, kActions> probs{};
  float z = 0.0f;
  for (int i = 0; i < kActions; ++i) {
    probs[i] = std::exp((logits[i] - max_logit) * inv_t);
    z += probs[i];
  }
  for (float& p : probs) p /= std::max(z, 1e-8f);

  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  float r = u(rng);
  float c = 0.0f;
  for (int i = 0; i < kActions; ++i) {
    c += probs[i];
    if (r <= c) return i;
  }
  return kActions - 1;
}

// ── Population Initialization ─────────────────────────────────────────────────
inline void init_population(std::vector<Agent>& population, const Config& cfg, std::mt19937_64& rng) {
  std::uniform_real_distribution<float> xdist(0.0f, static_cast<float>(cfg.width - 1));
  std::uniform_real_distribution<float> ydist(0.0f, static_cast<float>(cfg.height - 1));
  std::normal_distribution<float> gdist(0.0f, 0.35f);
  std::uniform_int_distribution<int> lifespan_dist(40, 220);

  population.resize(static_cast<size_t>(cfg.agents));
  const int predator_count = static_cast<int>(cfg.agents * cfg.predator_ratio);

  for (int idx = 0; idx < cfg.agents; ++idx) {
    Agent& a = population[idx];
    a.pos = {xdist(rng), ydist(rng)};
    a.energy = 6.0f;
    a.fitness = 0.0f;
    a.novelty_score = 0.0f;
    a.age = 0;
    a.alive = true;
    a.kills = 0;
    a.type = (idx < predator_count) ? AgentType::Predator : AgentType::Herbivore;
    a.max_lifespan = lifespan_dist(rng);
    a.metabolic_rate = 0.025f + 0.01f * (a.type == AgentType::Predator ? 1.0f : 0.0f);

    for (float& g : a.genome) {
      g = gdist(rng);
    }
    // Encode lifespan hint in a specific genome position (last few genes)
    a.genome[kGenomeSize - 1] = static_cast<float>(a.max_lifespan) / 220.0f;
  }
}

// ── Hunting Mechanic ─────────────────────────────────────────────────────────
inline void resolve_hunting(std::vector<Agent>& population, const Config& cfg,
                             WorldFields& world, std::mt19937_64& rng) {
  // Build spatial index: cell → list of agent indices
  const size_t cells = static_cast<size_t>(cfg.width) * static_cast<size_t>(cfg.height);
  std::vector<std::vector<int>> cell_agents(cells);
  for (int idx = 0; idx < static_cast<int>(population.size()); ++idx) {
    if (!population[idx].alive) continue;
    int x = static_cast<int>(population[idx].pos.x);
    int y = static_cast<int>(population[idx].pos.y);
    cell_agents[idx_2d(x, y, cfg)].push_back(idx);
  }

  std::uniform_real_distribution<float> roll(0.0f, 1.0f);

  for (int idx = 0; idx < static_cast<int>(population.size()); ++idx) {
    Agent& predator = population[idx];
    if (!predator.alive || predator.type != AgentType::Predator) continue;

    int px = static_cast<int>(predator.pos.x);
    int py = static_cast<int>(predator.pos.y);

    // Scan current + adjacent cells for prey
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = std::clamp(px + dx, 0, cfg.width - 1);
        int ny = std::clamp(py + dy, 0, cfg.height - 1);
        size_t ci = idx_2d(nx, ny, cfg);

        for (int prey_idx : cell_agents[ci]) {
          Agent& prey = population[prey_idx];
          if (!prey.alive || prey.type != AgentType::Herbivore) continue;

          // Hunt attempt
          float success_prob = cfg.hunt_success_prob;
          // Predators hunt better with more energy
          success_prob += 0.05f * (predator.energy / 10.0f);
          // Prey harder to catch if high energy (can flee)
          success_prob -= 0.05f * (prey.energy / 10.0f);
          success_prob = std::clamp(success_prob, 0.05f, 0.75f);

          if (roll(rng) < success_prob) {
            // Successful hunt: transfer energy
            float energy_gain = std::min(prey.energy * 0.6f, 3.0f);
            predator.energy += energy_gain;
            predator.fitness += 2.0f;
            predator.kills += 1;
            prey.energy -= energy_gain;
            prey.fitness -= 1.5f;

            // Predator deposits pheromone (danger signal)
            world.pheromone[ci] += 0.5f;

            // One kill per predator per step
            goto next_predator;
          }
        }
      }
    }
    // Failed hunt: stamina cost
    predator.energy -= 0.05f;
    next_predator:;
  }
}

// ── Step All Agents ──────────────────────────────────────────────────────────
inline void step_agents(std::vector<Agent>& population, const Config& cfg, WorldFields& world,
                         int step, std::mt19937_64& rng) {
  const float season_phase = std::sin((2.0f * kPi * static_cast<float>(step)) /
                                      std::max(40.0f, static_cast<float>(cfg.steps_per_generation)));

  for (auto& a : population) {
    if (!a.alive) continue;

    // ── Age & natural death ──
    a.age += 1;
    if (a.age > a.max_lifespan) {
      a.alive = false;
      // Corpse: deposit nutrients
      int cx = static_cast<int>(a.pos.x);
      int cy = static_cast<int>(a.pos.y);
      size_t ci = idx_2d(cx, cy, cfg);
      world.resources[ci] += a.energy * 0.3f;  // nutrient cycling
      continue;
    }

    int x = static_cast<int>(a.pos.x);
    int y = static_cast<int>(a.pos.y);
    size_t i = idx_2d(x, y, cfg);

    // Skip if in ocean
    if (static_cast<Biome>(world.biome[i]) == Biome::Ocean) {
      // Try to move out of ocean
      a.pos.x = std::clamp(a.pos.x + (a.pos.x < cfg.width / 2.0f ? 1.0f : -1.0f),
                            0.0f, static_cast<float>(cfg.width - 1));
      a.pos.y = std::clamp(a.pos.y + (a.pos.y < cfg.height / 2.0f ? 1.0f : -1.0f),
                            0.0f, static_cast<float>(cfg.height - 1));
      a.energy -= 0.1f;
      continue;
    }

    int action = sample_action(policy_logits(a, cfg, world, x, y, season_phase),
                               cfg.softmax_temperature, rng);

    float dx = 0.0f;
    float dy = 0.0f;
    if (action == 0) dy = -1.0f;
    if (action == 1) dy = 1.0f;
    if (action == 2) dx = -1.0f;
    if (action == 3) dx = 1.0f;

    // Check target cell — don't move into ocean
    float new_x = std::clamp(a.pos.x + dx, 0.0f, static_cast<float>(cfg.width - 1));
    float new_y = std::clamp(a.pos.y + dy, 0.0f, static_cast<float>(cfg.height - 1));
    size_t target_i = idx_2d(static_cast<int>(new_x), static_cast<int>(new_y), cfg);

    if (static_cast<Biome>(world.biome[target_i]) == Biome::Ocean) {
      // Can't move into ocean — stay put (effectively rest)
      dx = 0.0f;
      dy = 0.0f;
    } else {
      a.pos.x = new_x;
      a.pos.y = new_y;
    }

    x = static_cast<int>(a.pos.x);
    y = static_cast<int>(a.pos.y);
    i = idx_2d(x, y, cfg);

    world.occupancy[i] += 1.0f;
    world.visitation[i] = static_cast<uint16_t>(std::min(65535, world.visitation[i] + 1));

    // ── Movement cost (biome-dependent) ──
    Biome curBiome = static_cast<Biome>(world.biome[i]);
    const float biome_mult = biome_move_cost(curBiome);
    const float move_cost = (0.025f + 0.018f * (std::abs(dx) + std::abs(dy))) * biome_mult;
    const float thermal_penalty = 0.04f * std::abs(world.temperature[i] - 0.58f);

    // ── Aging metabolic cost: increases with age ──
    float age_ratio = static_cast<float>(a.age) / static_cast<float>(std::max(a.max_lifespan, 1));
    float metabolic_cost = a.metabolic_rate * (1.0f + 0.3f * age_ratio);

    // ── Resource harvest (herbivores only; predators get energy from hunting) ──
    float harvest = 0.0f;
    if (a.type == AgentType::Herbivore) {
      harvest = std::min(world.resources[i], 0.11f + 0.015f * (action == 4 ? 1.0f : 0.0f));
      world.resources[i] -= harvest;

      // Herbivore deposits "food-here" pheromone
      if (harvest > 0.05f) {
        world.pheromone[i] += 0.15f * harvest;
      }
    }

    // ── Intrinsic novelty reward ──
    const float novelty = 1.0f / std::sqrt(1.0f + static_cast<float>(world.visitation[i]));
    a.novelty_score += novelty;

    // ── Social reward (moderate density) ──
    const float density = mean3x3(world.occupancy, x, y, cfg);
    const float social = std::exp(-std::pow(density - 2.2f, 2.0f) / 3.0f) * 0.03f;

    // ── Reward aggregation ──
    const float reward = 0.9f * harvest + 0.06f * novelty + social
                        - move_cost - thermal_penalty - metabolic_cost;
    a.energy += reward;
    a.fitness += reward;

    if (a.energy < 0.0f) {
      a.fitness -= 1.0f;
      a.energy = 0.0f;
    }
  }

  // ── Predator hunting phase (after all movement) ──
  resolve_hunting(population, cfg, world, rng);
}

}  // namespace sim
