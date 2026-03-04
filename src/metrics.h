#pragma once
#include "types.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sim {

// ── Compute Metrics ──────────────────────────────────────────────────────────
inline Metrics compute_metrics(const std::vector<Agent>& population, const WorldFields& world,
                                const Config& cfg) {
  Metrics m;
  if (population.empty()) return m;

  std::unordered_map<int, int> species_counts;
  species_counts.reserve(128);

  float sum_f = 0.0f;
  float sum_n = 0.0f;
  float sum_age = 0.0f;
  float sum_lifespan = 0.0f;
  float best = -1e9f;
  int alive_count = 0;

  for (const auto& a : population) {
    sum_f += a.fitness;
    sum_n += a.novelty_score;
    sum_age += static_cast<float>(a.age);
    sum_lifespan += static_cast<float>(a.max_lifespan);
    best = std::max(best, a.fitness);

    if (a.alive) {
      ++alive_count;
      if (a.type == AgentType::Herbivore) ++m.herbivore_count;
      else ++m.predator_count;
    }
    species_counts[a.species_id] += 1;
  }

  const float n = static_cast<float>(population.size());
  m.mean_fitness = sum_f / n;
  m.mean_novelty = sum_n / n;
  m.best_fitness = best;
  m.mean_age = sum_age / n;
  m.mean_lifespan = sum_lifespan / n;
  m.alive_count = alive_count;

  // Shannon diversity
  float h = 0.0f;
  for (const auto& kv : species_counts) {
    float p = kv.second / n;
    if (p > 0.0f) h -= p * std::log(std::max(p, 1e-8f));
  }
  m.diversity_shannon = h;
  m.species_count = static_cast<int>(species_counts.size());

  // Total pheromone
  m.total_pheromone = 0.0f;
  for (float p : world.pheromone) m.total_pheromone += p;

  // Biome distribution
  const size_t cells = static_cast<size_t>(cfg.width) * static_cast<size_t>(cfg.height);
  std::array<int, 6> biome_counts{};
  for (size_t i = 0; i < cells; ++i) {
    int b = world.biome[i];
    if (b >= 0 && b < 6) biome_counts[b]++;
  }
  for (int i = 0; i < 6; ++i) {
    m.biome_distribution[i] = static_cast<float>(biome_counts[i]) / static_cast<float>(cells);
  }

  return m;
}

// ── Biome Name ───────────────────────────────────────────────────────────────
inline const char* biome_name(int b) {
  switch (b) {
    case 0: return "Ocean";
    case 1: return "Tundra";
    case 2: return "Desert";
    case 3: return "Grassland";
    case 4: return "Forest";
    case 5: return "Jungle";
    default: return "Unknown";
  }
}

// ── JSON Output ──────────────────────────────────────────────────────────────
inline std::string summary_json(const Config& cfg, const std::vector<Metrics>& metrics,
                                 const std::vector<SpeciesRecord>& species_records) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(4);
  os << "{\n";
  os << "  \"seed\": " << cfg.seed << ",\n";
  os << "  \"world\": {\"width\": " << cfg.width << ", \"height\": " << cfg.height << "},\n";
  os << "  \"agents\": " << cfg.agents << ",\n";
  os << "  \"config\": {"
     << "\"steps_per_generation\": " << cfg.steps_per_generation
     << ", \"softmax_temperature\": " << cfg.softmax_temperature
     << ", \"predator_ratio\": " << cfg.predator_ratio
     << ", \"hunt_success_prob\": " << cfg.hunt_success_prob
     << ", \"pheromone_decay\": " << cfg.pheromone_decay
     << ", \"speciation_threshold\": " << cfg.speciation_threshold
     << "},\n";

  // ── Generations ──
  os << "  \"generations\": [\n";
  for (size_t i = 0; i < metrics.size(); ++i) {
    const auto& m = metrics[i];
    os << "    {"
       << "\"generation\": " << i
       << ", \"mean_fitness\": " << m.mean_fitness
       << ", \"best_fitness\": " << m.best_fitness
       << ", \"mean_novelty\": " << m.mean_novelty
       << ", \"species_shannon\": " << m.diversity_shannon
       << ", \"species_count\": " << m.species_count
       << ", \"herbivore_count\": " << m.herbivore_count
       << ", \"predator_count\": " << m.predator_count
       << ", \"alive_count\": " << m.alive_count
       << ", \"mean_age\": " << m.mean_age
       << ", \"mean_lifespan\": " << m.mean_lifespan
       << ", \"total_pheromone\": " << m.total_pheromone
       << ", \"extinction_events\": " << m.extinction_events
       << ", \"speciation_events\": " << m.speciation_events
       << ", \"biome_distribution\": {";
    for (int b = 0; b < 6; ++b) {
      os << "\"" << biome_name(b) << "\": " << m.biome_distribution[b];
      if (b < 5) os << ", ";
    }
    os << "}";
    os << "}";
    if (i + 1 != metrics.size()) os << ",";
    os << "\n";
  }
  os << "  ],\n";

  // ── Species Records ──
  os << "  \"species_records\": [\n";
  for (size_t i = 0; i < species_records.size(); ++i) {
    const auto& sr = species_records[i];
    os << "    {"
       << "\"species_id\": " << sr.species_id
       << ", \"generation_born\": " << sr.generation_born
       << ", \"generation_extinct\": " << sr.generation_extinct
       << ", \"peak_population\": " << sr.peak_population
       << ", \"mean_fitness\": " << sr.mean_fitness
       << ", \"centroid\": [" << sr.centroid_genome[0]
       << ", " << sr.centroid_genome[1]
       << ", " << sr.centroid_genome[2]
       << ", " << sr.centroid_genome[3] << "]"
       << "}";
    if (i + 1 != species_records.size()) os << ",";
    os << "\n";
  }
  os << "  ]\n";
  os << "}\n";
  return os.str();
}

}  // namespace sim
