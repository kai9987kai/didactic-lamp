#pragma once
#include "types.h"
#include "world.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace sim {

inline float climate_match(const Agent& a, float temperature, float moisture) {
  const float thermal_match = 1.0f - std::abs(temperature - a.preferred_temperature);
  const float moisture_match = 1.0f - std::abs(moisture - a.preferred_moisture);
  return clamp_value(0.55f * thermal_match + 0.45f * moisture_match, 0.0f, 1.0f);
}

// ── Neural Policy Network (RNN) ──────────────────────────────────────────────
inline std::array<float, kOutputNodes> policy_logits(Agent& a, const Config& cfg,
                                                    const WorldFields& world, int x, int y,
                                                    float season_phase) {
  const size_t i = idx_2d(x, y, cfg);
  const float nx = a.pos.x / static_cast<float>(cfg.width - 1);
  const float ny = a.pos.y / static_cast<float>(cfg.height - 1);
  const float h = world.height[i];
  const float t = world.temperature[i];
  const float r = world.resources[i];
  const float e = a.energy / 12.0f; // normalized
  const float pheromone = std::min(world.pheromone[i], 5.0f) / 5.0f;  
  const float moist = world.moisture[i];
  const float biome_f = static_cast<float>(world.biome[i]) / 5.0f;    
  const float toxicity = world.toxicity[i];
  const float niche_match = climate_match(a, t, moist);
  const float niche_stress = 1.0f - niche_match;

  // Morphology-driven density
  const float density = density_radius(world.occupancy, x, y, a.sensory_radius, cfg);

  std::array<float, kInputFeatures> feat{nx, ny, h, t, r, density, e, season_phase,
                                         pheromone, moist, biome_f, toxicity, niche_match, niche_stress,
                                         a.memory[0], a.memory[1], a.memory[2], a.memory[3]};

  const int w1_size = kInputFeatures * kHiddenNeurons;
  const int b1_offset = w1_size;
  const int w2_offset = b1_offset + kHiddenNeurons;
  const int b2_offset = w2_offset + kHiddenNeurons * kOutputNodes;

  std::array<float, kHiddenNeurons> hidden{};
  for (int hid = 0; hid < kHiddenNeurons; ++hid) {
    float s = a.genome[b1_offset + hid];  // bias
    for (int inp = 0; inp < kInputFeatures; ++inp) {
      s += a.genome[hid * kInputFeatures + inp] * feat[inp];
    }
    hidden[hid] = std::tanh(s);
  }

  std::array<float, kOutputNodes> outputs{};
  for (int out = 0; out < kOutputNodes; ++out) {
    float s = a.genome[b2_offset + out];  // bias
    for (int hid = 0; hid < kHiddenNeurons; ++hid) {
      s += a.genome[w2_offset + out * kHiddenNeurons + hid] * hidden[hid];
    }
    outputs[out] = s;
  }
  
  // Update internal RNN memory state using the last 4 outputs (squashed to -1, 1)
  for(int m=0; m < kMemoryStates; ++m) {
    a.memory[m] = std::tanh(outputs[kActions + m]);
  }

  return outputs;
}

inline int sample_action(const std::array<float, kOutputNodes>& outputs, float temperature,
                          std::mt19937_64& rng) {
  const float inv_t = 1.0f / std::max(0.05f, temperature);
  
  // Look only at the first kActions logits
  float max_logit = -1e9f;
  for(int i=0; i<kActions; ++i) max_logit = std::max(max_logit, outputs[i]);

  std::array<float, kActions> probs{};
  float z = 0.0f;
  for (int i = 0; i < kActions; ++i) {
    probs[i] = std::exp((outputs[i] - max_logit) * inv_t);
    z += probs[i];
  }
  for (float& p : probs) p /= std::max(z, 1e-8f);

  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  float c = 0.0f, r = u(rng);
  for (int i = 0; i < kActions; ++i) {
    c += probs[i];
    if (r <= c) return i;
  }
  return kActions - 1;
}

// Extract morphology traits from specific genome indices
inline void decode_morphology(Agent& a) {
  // We use the very end of the genome array for morphology
  // Genes are naturally ~ N(0, 0.35). Convert to useful ranges.
  const int end = kGenomeSize;
  
  // Body Size: 0.5x to 2.0x
  a.body_size = clamp_value(1.0f + a.genome[end - 6], 0.5f, 2.0f);
  
  // Speed Modifier: 0.5x to 2.0x 
  a.speed_mod = clamp_value(1.0f + a.genome[end - 5], 0.5f, 2.0f);
  
  // Sensory Radius: 1.0 to 3.0 cells
  a.sensory_radius = clamp_value(2.0f + a.genome[end - 4], 1.0f, 3.0f);
  
  // Toxicity Resistance: 0.0 to 1.0 (requires tight adaptation)
  a.tox_resistance = clamp_value(0.5f + a.genome[end - 3], 0.0f, 1.0f);
  a.preferred_temperature = clamp_value(0.5f + 0.5f * a.genome[end - 2], 0.0f, 1.0f);
  a.preferred_moisture = clamp_value(0.5f + 0.5f * a.genome[end - 1], 0.0f, 1.0f);
  
  // Recalculate biological limits based on morphology
  // Huge, fast agents burn more energy!
  a.metabolic_rate = 0.008f 
                   * (a.body_size * a.body_size) // Quadratic mass penalty
                   * a.speed_mod;                // Linear speed penalty
                   
  if(a.type == AgentType::Predator) a.metabolic_rate *= 1.2f; // Carnivores burn hotter
}

// ── Population Initialization ─────────────────────────────────────────────────
inline void init_population(std::vector<Agent>& population, const Config& cfg, std::mt19937_64& rng) {
  std::uniform_real_distribution<float> xdist(0.0f, static_cast<float>(cfg.width - 1));
  std::uniform_real_distribution<float> ydist(0.0f, static_cast<float>(cfg.height - 1));
  std::normal_distribution<float> gdist(0.0f, 0.35f);
  std::uniform_int_distribution<int> lifespan_dist(200, 600); // Expanded lifespan
  std::uniform_int_distribution<int> gender_dist(0, 1);

  population.resize(static_cast<size_t>(cfg.initial_agents));
  const int predator_count = static_cast<int>(cfg.initial_agents * cfg.predator_ratio);

  for (int idx = 0; idx < cfg.initial_agents; ++idx) {
    Agent& a = population[idx];
    a.pos = {xdist(rng), ydist(rng)};
    a.energy = 12.0f;
    a.fitness = 0.0f;
    a.novelty_score = 0.0f;
    a.age = 0;
    a.birth_tick = 0;
    a.alive = true;
    a.kills = 0;
    
    a.type = (idx < predator_count) ? AgentType::Predator : AgentType::Herbivore;
    a.gender = static_cast<Gender>(gender_dist(rng));
    a.max_lifespan = lifespan_dist(rng);

    for (float& g : a.genome) g = gdist(rng);
    for (float& m : a.memory) m = 0.0f;

    decode_morphology(a);
  }
}

// ── Hunting Mechanic ─────────────────────────────────────────────────────────
inline void resolve_hunting(std::vector<Agent>& population, const Config& cfg,
                             WorldFields& world, std::mt19937_64& rng) {
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

    int r = static_cast<int>(std::ceil(predator.sensory_radius));
    for (int dy = -r; dy <= r; ++dy) {
      for (int dx = -r; dx <= r; ++dx) {
        if(dx*dx + dy*dy > predator.sensory_radius * predator.sensory_radius) continue;
        
        int nx = clamp_value(px + dx, 0, cfg.width - 1);
        int ny = clamp_value(py + dy, 0, cfg.height - 1);
        size_t ci = idx_2d(nx, ny, cfg);

        for (int prey_idx : cell_agents[ci]) {
          Agent& prey = population[prey_idx];
          if (!prey.alive || prey.type != AgentType::Herbivore) continue;

          // Body Size heavily impacts hunting!
          float size_advantage = predator.body_size / prey.body_size;
          float success_prob = 0.12f * size_advantage; // Much lower success
          success_prob += 0.05f * (predator.energy / 12.0f);
          success_prob = clamp_value(success_prob, 0.05f, 0.85f);

          if (roll(rng) < success_prob) {
            float energy_gain = std::min(prey.energy * 0.7f + (prey.body_size * 2.0f), 8.0f);
            predator.energy += energy_gain;
            predator.fitness += 2.0f;
            predator.kills += 1;
            
            prey.energy = 0.0f;
            prey.alive = false; // Killed instantly in Phase 2

            world.pheromone[ci] += 0.5f;
            goto next_predator;
          }
        }
      }
    }
    predator.energy -= 0.05f * predator.body_size; // Big predators starve faster if they fail
    next_predator:;
  }
}

// ── Step All Agents ──────────────────────────────────────────────────────────
inline void step_agents_movement(std::vector<Agent>& population, const Config& cfg, WorldFields& world,
                         int tick, std::mt19937_64& rng) {
  const float season_phase = std::sin((2.0f * kPi * static_cast<float>(tick)) / 300.0f);
  std::uniform_real_distribution<float> speed_roll(0.0f, 1.0f);

  for (auto& a : population) {
    if (!a.alive) continue;

    a.age += 1;
    if (a.age > a.max_lifespan) {
      a.alive = false;
      int cx = static_cast<int>(a.pos.x), cy = static_cast<int>(a.pos.y);
      world.resources[idx_2d(cx, cy, cfg)] += a.energy * 0.3f + a.body_size; 
      continue;
    }

    // Determine how many moves we get this tick (speed morphology)
    int moves = 1;
    if(a.speed_mod > 1.25f && speed_roll(rng) < (a.speed_mod - 1.0f) * 0.5f) moves = 2; // Extra fast
    if(a.speed_mod < 0.75f && speed_roll(rng) < (1.0f - a.speed_mod)) moves = 0;        // Sluggish

    for(int m=0; m < moves && a.alive; ++m) {
      int x = static_cast<int>(a.pos.x);
      int y = static_cast<int>(a.pos.y);
      size_t i = idx_2d(x, y, cfg);

      if (static_cast<Biome>(world.biome[i]) == Biome::Ocean) {
        a.pos.x = clamp_value(a.pos.x + (a.pos.x < cfg.width / 2.0f ? 1.0f : -1.0f), 0.0f, static_cast<float>(cfg.width - 1));
        a.pos.y = clamp_value(a.pos.y + (a.pos.y < cfg.height / 2.0f ? 1.0f : -1.0f), 0.0f, static_cast<float>(cfg.height - 1));
        a.energy -= 0.3f * a.body_size; // Oceans are deadly
        if(a.energy <= 0) a.alive = false;
        continue;
      }

      int action = sample_action(policy_logits(a, cfg, world, x, y, season_phase), cfg.softmax_temperature, rng);

      float dx = 0.0f, dy = 0.0f;
      if (action == 0) dy = -1.0f;
      if (action == 1) dy = 1.0f;
      if (action == 2) dx = -1.0f;
      if (action == 3) dx = 1.0f;

      float new_x = clamp_value(a.pos.x + dx, 0.0f, static_cast<float>(cfg.width - 1));
      float new_y = clamp_value(a.pos.y + dy, 0.0f, static_cast<float>(cfg.height - 1));
      size_t target_i = idx_2d(static_cast<int>(new_x), static_cast<int>(new_y), cfg);

      if (static_cast<Biome>(world.biome[target_i]) != Biome::Ocean) {
        a.pos.x = new_x; a.pos.y = new_y;
      }

      x = static_cast<int>(a.pos.x); y = static_cast<int>(a.pos.y);
      i = idx_2d(x, y, cfg);

      world.occupancy[i] += a.body_size; // Big agents take more space!

      Biome curBiome = static_cast<Biome>(world.biome[i]);
      const float biome_mult = biome_move_cost(curBiome);
      const float move_cost = (0.015f + 0.01f * (std::abs(dx) + std::abs(dy))) * biome_mult * a.body_size;
      const float thermal_penalty = 0.04f * std::abs(world.temperature[i] - 0.58f) * (2.0f - a.body_size); // Small agents get colder
      const float habitat_match_score = climate_match(a, world.temperature[i], world.moisture[i]);
      const float habitat_bonus = 0.03f * habitat_match_score;
      const float habitat_stress = 0.08f * std::pow(1.0f - habitat_match_score, 2.0f);

      float age_ratio = static_cast<float>(a.age) / static_cast<float>(std::max(a.max_lifespan, 1));
      float metabolic_cost = a.metabolic_rate * (1.0f + 0.3f * age_ratio);

      // Phase 2: Botany Toxicity Damage vs Resistance
      float tox_damage = 0.0f;
      float diff = world.toxicity[i] - a.tox_resistance;
      if(diff > 0.1f) {
        tox_damage = diff * 0.15f; // Constant drain if in toxic biome without resistance
      }

      float harvest = 0.0f;
      if (a.type == AgentType::Herbivore) {
        // Must spend action 4 (Rest/Eat) to harvest deeply
        float harvest_cap = (action == 4) ? (0.25f * a.body_size) : (0.08f * a.body_size);
        harvest = std::min(world.resources[i], harvest_cap);
        
        // Eating toxic plants increases the damage!
        if(diff > 0.0f) tox_damage += diff * harvest * 0.5f;

        world.resources[i] -= harvest;
        if (harvest > 0.05f) world.pheromone[i] += 0.15f * harvest;
      }

      world.visitation[i] += 1;
      const float novelty = 1.0f / std::sqrt(1.0f + static_cast<float>(world.visitation[i]));
      a.novelty_score += novelty;

      const float density = density_radius(world.occupancy, x, y, 1.5f, cfg);
      const float social = std::exp(-std::pow(density - 2.5f, 2.0f) / 3.0f) * 0.03f;

      const float reward = 0.9f * harvest + 0.06f * novelty + social + habitat_bonus
                         - move_cost - thermal_penalty - habitat_stress - metabolic_cost - tox_damage;
      
      a.energy += reward;
      a.fitness += reward;

      if (a.energy <= 0.0f) {
        a.alive = false; // Starved or poisoned!
      }
    }
  }

  resolve_hunting(population, cfg, world, rng);
}

}  // namespace sim
