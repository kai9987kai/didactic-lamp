#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace sim {

// ── Constants ────────────────────────────────────────────────────────────────
constexpr int kActions = 5;  // up, down, left, right, rest
constexpr float kPi = 3.14159265358979323846f;

// Neural policy architecture: 11 inputs → 8 hidden (tanh) → 5 outputs
constexpr int kInputFeatures = 11;   // nx,ny,h,t,r,density,e,season,pheromone,moisture,biome
constexpr int kHiddenNeurons = 8;
constexpr int kGenomeSize = kInputFeatures * kHiddenNeurons   // input→hidden weights
                          + kHiddenNeurons                     // hidden biases
                          + kHiddenNeurons * kActions           // hidden→output weights
                          + kActions;                           // output biases
// = 11*8 + 8 + 8*5 + 5 = 88 + 8 + 40 + 5 = 141

// ── Enums ────────────────────────────────────────────────────────────────────
enum class AgentType : uint8_t { Herbivore = 0, Predator = 1 };

enum class Biome : uint8_t {
  Ocean = 0,
  Tundra = 1,
  Desert = 2,
  Grassland = 3,
  Forest = 4,
  Jungle = 5
};

// ── Core Structs ─────────────────────────────────────────────────────────────
struct Vec2 {
  float x{};
  float y{};
};

struct Agent {
  Vec2 pos{};
  float energy{6.0f};
  float fitness{0.0f};
  float novelty_score{0.0f};
  int age{0};
  int species_id{0};
  AgentType type{AgentType::Herbivore};
  float metabolic_rate{0.03f};
  int max_lifespan{160};
  bool alive{true};
  int kills{0};
  std::array<float, kGenomeSize> genome{};
};

struct Config {
  int width{128};
  int height{128};
  int agents{4096};
  int generations{20};
  int steps_per_generation{220};
  uint64_t seed{7};
  float softmax_temperature{0.8f};
  float predator_ratio{0.2f};        // fraction of population that starts as predators
  float hunt_success_prob{0.35f};     // base probability of successful hunt
  float pheromone_decay{0.92f};       // per-step decay multiplier
  float speciation_threshold{0.80f};  // genetic distance for new species
};

struct WorldFields {
  std::vector<float> height;
  std::vector<float> temperature;
  std::vector<float> resources;
  std::vector<float> occupancy;
  std::vector<uint16_t> visitation;
  std::vector<float> pheromone;
  std::vector<float> moisture;
  std::vector<uint8_t> biome;  // Biome enum stored as uint8_t
};

struct SpeciesRecord {
  int species_id{};
  int generation_born{};
  int generation_extinct{-1};  // -1 = still alive
  int peak_population{0};
  float mean_fitness{0.0f};
  std::array<float, 4> centroid_genome{};  // first 4 genome values as signature
};

struct Metrics {
  float mean_fitness{};
  float best_fitness{};
  float mean_novelty{};
  float diversity_shannon{};
  int herbivore_count{};
  int predator_count{};
  int species_count{};
  int extinction_events{};
  int speciation_events{};
  float mean_age{};
  float mean_lifespan{};
  float total_pheromone{};
  std::array<float, 6> biome_distribution{};  // fraction of land per biome type
  int alive_count{};
};

// ── Helpers ──────────────────────────────────────────────────────────────────
inline size_t idx_2d(int x, int y, const Config& cfg) {
  x = std::clamp(x, 0, cfg.width - 1);
  y = std::clamp(y, 0, cfg.height - 1);
  return static_cast<size_t>(y) * static_cast<size_t>(cfg.width) + static_cast<size_t>(x);
}

inline float mean3x3(const std::vector<float>& field, int x, int y, const Config& cfg) {
  float sum = 0.0f;
  int count = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      sum += field[idx_2d(x + dx, y + dy, cfg)];
      ++count;
    }
  }
  return sum / static_cast<float>(count);
}

}  // namespace sim
