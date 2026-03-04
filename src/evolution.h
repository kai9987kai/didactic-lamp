#pragma once
#include "types.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace sim {

// ── Genetic Distance ─────────────────────────────────────────────────────────
inline float genetic_distance(const Agent& a, const Agent& b) {
  float sum = 0.0f;
  const int compare_len = std::min(32, kGenomeSize);
  for (int i = 0; i < compare_len; ++i) {
    float d = a.genome[i] - b.genome[i];
    sum += d * d;
  }
  return std::sqrt(sum / static_cast<float>(compare_len));
}

// ── Species Classification with Centroids ────────────────────────────────────
struct SpeciesTracker {
  struct Centroid {
    int id;
    std::array<float, 32> genome_sig{};
    int population{0};
  };

  std::vector<Centroid> centroids;
  int next_species_id{0};
  std::vector<SpeciesRecord> records;

  void classify(std::vector<Agent>& population, float threshold, int tick) {
    for (auto& c : centroids) c.population = 0;

    for (auto& a : population) {
      if (!a.alive) continue;

      float best_dist = 1e9f;
      int best_id = -1;

      for (size_t ci = 0; ci < centroids.size(); ++ci) {
        float dist = 0.0f;
        const int compare_len = std::min(32, kGenomeSize);
        for (int i = 0; i < compare_len; ++i) {
          float d = a.genome[i] - centroids[ci].genome_sig[i];
          dist += d * d;
        }
        dist = std::sqrt(dist / static_cast<float>(compare_len));

        if (dist < best_dist) {
          best_dist = dist;
          best_id = static_cast<int>(ci);
        }
      }

      if (best_dist < threshold && best_id >= 0) {
        a.species_id = centroids[best_id].id;
        centroids[best_id].population++;
        const int compare_len = std::min(32, kGenomeSize);
        float alpha = 0.01f;
        for (int i = 0; i < compare_len; ++i) {
          centroids[best_id].genome_sig[i] =
              (1.0f - alpha) * centroids[best_id].genome_sig[i] + alpha * a.genome[i];
        }
      } else {
        Centroid new_c;
        new_c.id = next_species_id++;
        new_c.population = 1;
        const int compare_len = std::min(32, kGenomeSize);
        for (int i = 0; i < compare_len; ++i) new_c.genome_sig[i] = a.genome[i];
        centroids.push_back(new_c);
        a.species_id = new_c.id;

        SpeciesRecord rec;
        rec.species_id = new_c.id;
        rec.tick_born = tick;
        rec.tick_extinct = -1;
        for (int i = 0; i < 4 && i < kGenomeSize; ++i) rec.centroid_genome[i] = a.genome[i];
        records.push_back(rec);
      }
    }

    for (auto it = centroids.begin(); it != centroids.end();) {
      if (it->population == 0) {
        for (auto& r : records) {
          if (r.species_id == it->id && r.tick_extinct < 0) {
            r.tick_extinct = tick;
            break;
          }
        }
        it = centroids.erase(it);
      } else {
        for (auto& r : records) {
          if (r.species_id == it->id) {
            r.peak_population = std::max(r.peak_population, it->population);
            break;
          }
        }
        ++it;
      }
    }
  }

  int count_extinctions_since(int last_tick, int current_tick) const {
    int count = 0;
    for (const auto& r : records) {
      if (r.tick_extinct >= last_tick && r.tick_extinct <= current_tick) ++count;
    }
    return count;
  }

  int count_speciations_since(int last_tick, int current_tick) const {
    int count = 0;
    for (const auto& r : records) {
      if (r.tick_born >= last_tick && r.tick_born <= current_tick) ++count;
    }
    return count;
  }
};

// ── Continuous Sexual Reproduction (Proximity Async Mating) ────────────────
inline int resolve_mating(std::vector<Agent>& population, const Config& cfg,
                            std::mt19937_64& rng, int current_tick) {
  int births_this_tick = 0;
  
  // Build spatial index
  const size_t cells = static_cast<size_t>(cfg.width) * static_cast<size_t>(cfg.height);
  std::vector<std::vector<int>> cell_agents(cells);
  
  int curr_pop_size = 0;
  for (int idx = 0; idx < static_cast<int>(population.size()); ++idx) {
    if (!population[idx].alive) continue;
    curr_pop_size++;
    int x = static_cast<int>(population[idx].pos.x);
    int y = static_cast<int>(population[idx].pos.y);
    cell_agents[idx_2d(x, y, cfg)].push_back(idx);
  }

  // Cap population to prevent OOM
  if (curr_pop_size >= cfg.max_agents) return 0;

  std::uniform_real_distribution<float> coin(0.0f, 1.0f);
  std::normal_distribution<float> mut(0.0f, 0.04f); // Slightly higher mutation for continuous runs
  std::uniform_int_distribution<int> gender_dist(0, 1);
  std::uniform_int_distribution<int> lifespan_dist(200, 600);

  // Identify eligible parents
  for (int idx = 0; idx < static_cast<int>(population.size()); ++idx) {
    Agent& a = population[idx];
    if (!a.alive || a.energy < cfg.reproduction_threshold) continue;
    if (current_tick - a.last_mate_tick < 30) continue; // 30-tick cooldown

    int px = static_cast<int>(a.pos.x);
    int py = static_cast<int>(a.pos.y);

    // Scan adjacent for mate (radius 2)
    for (int dy = -2; dy <= 2; ++dy) {
      for (int dx = -2; dx <= 2; ++dx) {
        int nx = std::clamp(px + dx, 0, cfg.width - 1);
        int ny = std::clamp(py + dy, 0, cfg.height - 1);
        size_t ci = idx_2d(nx, ny, cfg);

        for (int mate_idx : cell_agents[ci]) {
          if (mate_idx == idx) continue;
          Agent& mate = population[mate_idx];
          
          if (!mate.alive || mate.energy < cfg.reproduction_threshold) continue;
          
          if (mate.species_id == a.species_id && 
              mate.type == a.type && 
              mate.gender != a.gender &&
              (current_tick - mate.last_mate_tick > 30)) {

            // Found a valid mate! Both lose energy and trigger cooldown.
            a.energy -= 5.0f;
            mate.energy -= 5.0f;
            a.last_mate_tick = current_tick;
            mate.last_mate_tick = current_tick;

            // Spawn Child
            Agent child;
            child.pos = {std::clamp(a.pos.x + (coin(rng)-0.5f), 0.0f, static_cast<float>(cfg.width - 1)),
                         std::clamp(a.pos.y + (coin(rng)-0.5f), 0.0f, static_cast<float>(cfg.height - 1))};
                         
            child.energy = 4.0f; // Started strong
            child.max_lifespan = lifespan_dist(rng);
            child.type = a.type;
            child.gender = static_cast<Gender>(gender_dist(rng));
            child.birth_tick = current_tick;
            
            for (float& m : child.memory) m = 0.0f;

            // Crossover
            for (size_t g = 0; g < kGenomeSize; ++g) {
              float inherited = (coin(rng) < 0.5f) ? a.genome[g] : mate.genome[g];
              if (coin(rng) < 0.2f) inherited += mut(rng); // Mutation rate
              child.genome[g] = inherited;
            }
            
            // Recalculate child morphology based on specific genes
            extern void decode_morphology(Agent&);
            decode_morphology(child);

            population.push_back(std::move(child));
            births_this_tick++;
            
            // Prevent multiple matings per tick
            goto next_agent; 
          }
        }
      }
    }
  next_agent:;
  }

  return births_this_tick;
}

// ── Clean Up Dead Agents ───────────────────────────────────────────────────
inline int cull_dead_agents(std::vector<Agent>& population) {
  int deaths = 0;
  auto it = std::remove_if(population.begin(), population.end(), 
                           [&deaths](const Agent& a) { 
                             if(!a.alive) deaths++;
                             return !a.alive; 
                           });
  population.erase(it, population.end());
  return deaths;
}

}  // namespace sim
