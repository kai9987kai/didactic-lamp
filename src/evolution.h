#pragma once
#include "types.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

namespace sim {

// ── Genetic Distance ─────────────────────────────────────────────────────────
inline float genetic_distance(const Agent& a, const Agent& b) {
  float sum = 0.0f;
  // Use first 32 genome values for efficient distance computation
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
    int generation_born{0};
  };

  std::vector<Centroid> centroids;
  int next_species_id{0};
  std::vector<SpeciesRecord> records;

  void classify(std::vector<Agent>& population, float threshold, int generation) {
    // Reset population counts
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
        // Update centroid with running average
        const int compare_len = std::min(32, kGenomeSize);
        float alpha = 0.01f;
        for (int i = 0; i < compare_len; ++i) {
          centroids[best_id].genome_sig[i] =
              (1.0f - alpha) * centroids[best_id].genome_sig[i] + alpha * a.genome[i];
        }
      } else {
        // New species!
        Centroid new_c;
        new_c.id = next_species_id++;
        new_c.population = 1;
        new_c.generation_born = generation;
        const int compare_len = std::min(32, kGenomeSize);
        for (int i = 0; i < compare_len; ++i) {
          new_c.genome_sig[i] = a.genome[i];
        }
        centroids.push_back(new_c);
        a.species_id = new_c.id;

        // Record new species
        SpeciesRecord rec;
        rec.species_id = new_c.id;
        rec.generation_born = generation;
        for (int i = 0; i < 4 && i < kGenomeSize; ++i) {
          rec.centroid_genome[i] = a.genome[i];
        }
        records.push_back(rec);
      }
    }

    // Track extinctions
    for (auto it = centroids.begin(); it != centroids.end();) {
      if (it->population == 0) {
        // Mark species as extinct in records
        for (auto& r : records) {
          if (r.species_id == it->id && r.generation_extinct < 0) {
            r.generation_extinct = generation;
            break;
          }
        }
        it = centroids.erase(it);
      } else {
        // Update peak population
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

  int count_extinctions(int generation) const {
    int count = 0;
    for (const auto& r : records) {
      if (r.generation_extinct == generation) ++count;
    }
    return count;
  }

  int count_speciations(int generation) const {
    int count = 0;
    for (const auto& r : records) {
      if (r.generation_born == generation) ++count;
    }
    return count;
  }
};

// ── Evolution ────────────────────────────────────────────────────────────────
inline void evolve(std::vector<Agent>& population, const Config& cfg, std::mt19937_64& rng) {
  // Sort by fitness + novelty (combined score)
  std::sort(population.begin(), population.end(), [](const Agent& a, const Agent& b) {
    float sa = a.fitness + 0.1f * a.novelty_score + (a.alive ? 0.0f : -5.0f);
    float sb = b.fitness + 0.1f * b.novelty_score + (b.alive ? 0.0f : -5.0f);
    return sa > sb;
  });

  // Separate elites by type
  const int elite_count = std::max(4, cfg.agents / 12);
  std::vector<const Agent*> herb_elites, pred_elites;
  for (int i = 0; i < static_cast<int>(population.size()) && 
       (static_cast<int>(herb_elites.size()) < elite_count || 
        static_cast<int>(pred_elites.size()) < elite_count); ++i) {
    if (population[i].type == AgentType::Herbivore && 
        static_cast<int>(herb_elites.size()) < elite_count) {
      herb_elites.push_back(&population[i]);
    } else if (population[i].type == AgentType::Predator && 
               static_cast<int>(pred_elites.size()) < elite_count) {
      pred_elites.push_back(&population[i]);
    }
  }

  // Ensure minimum pool size
  if (herb_elites.empty()) herb_elites.push_back(&population[0]);
  if (pred_elites.empty()) pred_elites.push_back(&population[0]);

  std::uniform_real_distribution<float> coin(0.0f, 1.0f);
  std::normal_distribution<float> mut(0.0f, 0.035f);
  std::uniform_real_distribution<float> xdist(0.0f, static_cast<float>(cfg.width - 1));
  std::uniform_real_distribution<float> ydist(0.0f, static_cast<float>(cfg.height - 1));
  std::uniform_int_distribution<int> lifespan_dist(40, 220);

  const int predator_target = static_cast<int>(cfg.agents * cfg.predator_ratio);

  std::vector<Agent> next(static_cast<size_t>(cfg.agents));
  for (int idx = 0; idx < cfg.agents; ++idx) {
    Agent& child = next[idx];
    child.type = (idx < predator_target) ? AgentType::Predator : AgentType::Herbivore;

    // Select parent pool based on type
    const auto& elites = (child.type == AgentType::Predator) ? pred_elites : herb_elites;
    std::uniform_int_distribution<int> pick(0, static_cast<int>(elites.size()) - 1);

    const Agent& p1 = *elites[pick(rng)];
    const Agent& p2 = *elites[pick(rng)];

    // ── Speciation-aware crossover: prefer genetically close parents ──
    for (size_t i = 0; i < child.genome.size(); ++i) {
      float inherited = (coin(rng) < 0.5f) ? p1.genome[i] : p2.genome[i];

      // Adaptive mutation: higher if parents underperform
      float adapt = 1.0f + std::max(0.0f, 0.5f - 0.5f * (p1.fitness + p2.fitness) / 100.0f);
      if (coin(rng) < 0.18f) inherited += mut(rng) * adapt;

      child.genome[i] = inherited;
    }

    child.pos = {xdist(rng), ydist(rng)};
    child.energy = 6.0f;
    child.fitness = 0.0f;
    child.novelty_score = 0.0f;
    child.age = 0;
    child.alive = true;
    child.kills = 0;
    child.max_lifespan = lifespan_dist(rng);
    child.metabolic_rate = 0.025f + 0.01f * (child.type == AgentType::Predator ? 1.0f : 0.0f);
  }

  population.swap(next);
}

}  // namespace sim
