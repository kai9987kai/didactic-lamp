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

inline float habitat_match_for_metrics(const Agent& a, const WorldFields& world, const Config& cfg) {
  const size_t i = idx_2d(static_cast<int>(a.pos.x), static_cast<int>(a.pos.y), cfg);
  const float thermal_match = 1.0f - std::abs(world.temperature[i] - a.preferred_temperature);
  const float moisture_match = 1.0f - std::abs(world.moisture[i] - a.preferred_moisture);
  return clamp_value(0.55f * thermal_match + 0.45f * moisture_match, 0.0f, 1.0f);
}

// ── Compute Continuous Metrics ───────────────────────────────────────────────
inline Metrics compute_metrics(const std::vector<Agent>& population, const WorldFields& world,
                                const Config& cfg, int current_tick, int births, int deaths) {
  Metrics m;
  m.tick = current_tick;
  m.births = births;
  m.deaths = deaths;
  const ActiveWorldEvent active_event = current_world_event(cfg, current_tick);
  m.active_event = active_event.type;
  m.event_intensity = active_event.intensity;

  if (population.empty()) return m;

  std::unordered_map<int, int> species_counts;
  species_counts.reserve(128);

  float sum_f = 0.0f;
  float max_f = -1e9f;
  float sum_n = 0.0f;
  float sum_age = 0.0f;
  float sum_size = 0.0f;
  float sum_speed = 0.0f;
  float sum_tox = 0.0f;
  float sum_match = 0.0f;

  for (const auto& a : population) {
    if (!a.alive) continue;
    sum_f += a.fitness;
    max_f = std::max(max_f, a.fitness);
    sum_n += a.novelty_score;
    sum_age += static_cast<float>(a.age);
    
    // Mean morphological traits
    sum_size += a.body_size;
    sum_speed += a.speed_mod;
    sum_tox += a.tox_resistance;
    sum_match += habitat_match_for_metrics(a, world, cfg);

    if (a.type == AgentType::Herbivore) ++m.herbivore_count;
    else ++m.predator_count;
    
    species_counts[a.species_id] += 1;
  }

  const float n = static_cast<float>(population.size());
  if(n > 0.0f) {
    m.mean_fitness = sum_f / n;
    m.max_fitness = max_f;
    m.mean_novelty = sum_n / n;
    m.mean_age = sum_age / n;
    m.mean_size = sum_size / n;
    m.mean_speed = sum_speed / n;
    m.mean_tox_res = sum_tox / n;
    m.mean_habitat_match = sum_match / n;
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
  float resource_sum = 0.0f;
  float toxicity_sum = 0.0f;
  for (size_t i = 0; i < cells; ++i) {
    int b = world.biome[i];
    if (b >= 0 && b < 6) biome_counts[b]++;
    resource_sum += world.resources[i];
    toxicity_sum += world.toxicity[i];
  }
  for (int i = 0; i < 6; ++i) {
    m.biome_distribution[i] = static_cast<float>(biome_counts[i]) / static_cast<float>(cells);
  }
  if (cells > 0) {
    m.mean_resources = resource_sum / static_cast<float>(cells);
    m.mean_toxicity = toxicity_sum / static_cast<float>(cells);
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
     << ", \"reproductive_distance\": " << cfg.reproductive_distance
     << ", \"shock_interval\": " << cfg.shock_interval
     << ", \"shock_duration\": " << cfg.shock_duration
     << ", \"shock_strength\": " << cfg.shock_strength
     << "},\n";

  os << "  \"ticks\": [\n";
  for (size_t i = 0; i < metrics.size(); ++i) {
    const auto& m = metrics[i];
    os << "    {"
       << "\"tick\": " << m.tick
       << ", \"mean_fitness\": " << m.mean_fitness
       << ", \"max_fitness\": " << m.max_fitness
       << ", \"species_shannon\": " << m.diversity_shannon
       << ", \"species_count\": " << m.species_count
       << ", \"herbivore_count\": " << m.herbivore_count
       << ", \"predator_count\": " << m.predator_count
       << ", \"deaths\": " << m.deaths
       << ", \"births\": " << m.births
        << ", \"mean_age\": " << m.mean_age
        << ", \"total_pheromone\": " << m.total_pheromone
        << ", \"mean_habitat_match\": " << m.mean_habitat_match
        << ", \"mean_resources\": " << m.mean_resources
        << ", \"mean_toxicity\": " << m.mean_toxicity
        << ", \"active_event\": \"" << world_event_name(m.active_event) << "\""
        << ", \"event_intensity\": " << m.event_intensity
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
