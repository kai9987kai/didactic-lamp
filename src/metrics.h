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

// ── Compute Continuous Metrics ───────────────────────────────────────────────
inline Metrics compute_metrics(const std::vector<Agent>& population, const WorldFields& world,
                                const Config& cfg, int current_tick, int births, int deaths) {
  Metrics m;
  m.tick = current_tick;
  m.births = births;
  m.deaths = deaths;

  if (population.empty()) return m;

  std::unordered_map<int, int> species_counts;
  species_counts.reserve(128);

  float sum_f = 0.0f;
  float sum_n = 0.0f;
  float sum_age = 0.0f;
  float sum_size = 0.0f;
  float sum_speed = 0.0f;
  float sum_tox = 0.0f;

  for (const auto& a : population) {
    if (!a.alive) continue;
    sum_f += a.fitness;
    sum_n += a.novelty_score;
    sum_age += static_cast<float>(a.age);
    
    // Mean morphological traits
    sum_size += a.body_size;
    sum_speed += a.speed_mod;
    sum_tox += a.tox_resistance;

    if (a.type == AgentType::Herbivore) ++m.herbivore_count;
    else ++m.predator_count;
    
    species_counts[a.species_id] += 1;
  }

  const float n = static_cast<float>(population.size());
  if(n > 0.0f) {
    m.mean_fitness = sum_f / n;
    m.mean_novelty = sum_n / n;
    m.mean_age = sum_age / n;
    m.mean_size = sum_size / n;
    m.mean_speed = sum_speed / n;
    m.mean_tox_res = sum_tox / n;
  }

  float h = 0.0f;
  for (const auto& kv : species_counts) {
    float p = kv.second / n;
    if (p > 0.0f) h -= p * std::log(std::max(p, 1e-8f));
  }
  m.diversity_shannon = h;
  m.species_count = static_cast<int>(species_counts.size());

  m.total_pheromone = 0.0f;
  for (float p : world.pheromone) m.total_pheromone += p;

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
  os << "  \"config\": {"
     << "\"simulation_ticks\": " << cfg.simulation_ticks
     << ", \"tick_interval\": " << cfg.snapshot_interval
     << ", \"reproduction_threshold\": " << cfg.reproduction_threshold
     << ", \"predator_ratio\": " << cfg.predator_ratio
     << ", \"speciation_threshold\": " << cfg.speciation_threshold
     << "},\n";

  os << "  \"ticks\": [\n";
  for (size_t i = 0; i < metrics.size(); ++i) {
    const auto& m = metrics[i];
    os << "    {"
       << "\"tick\": " << m.tick
       << ", \"mean_fitness\": " << m.mean_fitness
       << ", \"species_shannon\": " << m.diversity_shannon
       << ", \"species_count\": " << m.species_count
       << ", \"herbivore_count\": " << m.herbivore_count
       << ", \"predator_count\": " << m.predator_count
       << ", \"deaths\": " << m.deaths
       << ", \"births\": " << m.births
       << ", \"mean_age\": " << m.mean_age
       << ", \"total_pheromone\": " << m.total_pheromone
       << ", \"morphology\": {"
       << "\"mean_size\": " << m.mean_size
       << ", \"mean_speed\": " << m.mean_speed
       << ", \"mean_tox_res\": " << m.mean_tox_res
       << "}"
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

  os << "  \"species_records\": [\n";
  for (size_t i = 0; i < species_records.size(); ++i) {
    const auto& sr = species_records[i];
    os << "    {"
       << "\"species_id\": " << sr.species_id
       << ", \"tick_born\": " << sr.tick_born
       << ", \"tick_extinct\": " << sr.tick_extinct
       << ", \"peak_population\": " << sr.peak_population
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
