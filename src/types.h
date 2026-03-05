#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace sim {

// ── Constants ────────────────────────────────────────────────────────────────
constexpr int kActions = 5;  // up, down, left, right, rest
constexpr int kMemoryStates = 4; // RNN hidden state passed to next tick
constexpr int kOutputNodes = kActions + kMemoryStates; // 9 outputs
constexpr float kPi = 3.14159265358979323846f;

// Neural policy architecture: 16 inputs → 10 hidden (tanh) → 9 outputs
constexpr int kInputFeatures = 18;   // env state, niche match, and 4 recurrent memory values
constexpr int kHiddenNeurons = 10;
constexpr int kTraitGenes = 6;       // morphology + thermal/moisture niche preferences

constexpr int kBaseGenomeSize = kInputFeatures * kHiddenNeurons   // input→hidden weights
                              + kHiddenNeurons                     // hidden biases
                              + kHiddenNeurons * kOutputNodes      // hidden→output weights
                              + kOutputNodes;                      // output biases
// = 16*10 + 10 + 10*9 + 9 = 160 + 10 + 90 + 9 = 269
constexpr int kGenomeSize = kBaseGenomeSize + kTraitGenes;          // 295

// ── Enums ────────────────────────────────────────────────────────────────────
enum class AgentType : uint8_t { Herbivore = 0, Predator = 1 };
enum class Gender : uint8_t { Male = 0, Female = 1 };

enum class Biome : uint8_t {
  Ocean = 0,
  Tundra = 1,
  Desert = 2,
  Grassland = 3,
  Forest = 4,
  Jungle = 5
};

enum class WorldEvent : uint8_t {
  None = 0,
  Drought = 1,
  ColdSnap = 2,
  Bloom = 3,
  ToxicBloom = 4
};

// ── Core Structs ─────────────────────────────────────────────────────────────
struct Vec2 {
  float x{};
  float y{};
};

struct Agent {
  Vec2 pos{};
  float energy{8.0f};           // start with more reserve for mating
  float fitness{0.0f};
  float novelty_score{0.0f};
  int age{0};                   // discrete ticks alive
  int birth_tick{0};            // absolute simulation tick born
  
  int species_id{0};
  AgentType type{AgentType::Herbivore};
  Gender gender{Gender::Female};
  
  bool alive{true};
  int kills{0};
  int last_mate_tick{-100};      // Cooldown for reproduction
  
  // Recurrent Memory State
  std::array<float, kMemoryStates> memory{};
  
  // Morphology (Decoded from genome)
  float body_size{1.0f};         // 0.5 to 2.0
  float speed_mod{1.0f};         // 0.5 to 2.0 (chance to double-move or rest)
  float sensory_radius{1.5f};    // 1.0 to 3.0
  float tox_resistance{0.5f};    // 0.0 to 1.0
  float preferred_temperature{0.5f};
  float preferred_moisture{0.5f};
  
  // Biological Limits
  float metabolic_rate{0.03f};
  int max_lifespan{400};         // expanded for continuous simulation
  
  std::array<float, kGenomeSize> genome{};
};

struct Config {
  int width{128};
  int height{128};
  int initial_agents{2048};
  int max_agents{6000};              // population cap to prevent OOM
  int simulation_ticks{5000};        // Phase 2: continuous run length
  int snapshot_interval{100};        // Save metrics every N ticks
  uint64_t seed{7};
  float softmax_temperature{0.8f};
  float predator_ratio{0.05f};        
  float hunt_success_prob{0.35f};     
  float pheromone_decay{0.92f};       
  float speciation_threshold{0.55f};
  float reproductive_distance{0.42f};
  float reproduction_threshold{11.0f}; // Energy required to spawn offspring
  int shock_interval{180};
  int shock_duration{45};
  float shock_strength{0.18f};
};

struct ActiveWorldEvent {
  WorldEvent type{WorldEvent::None};
  float intensity{0.0f};
  float phase{0.0f};
  bool active{false};
};

struct WorldFields {
  std::vector<float> height;
  std::vector<float> temperature;
  std::vector<float> resources;
  std::vector<float> toxicity;       // Botany evolution: toxic flora
  std::vector<float> occupancy;
  std::vector<uint32_t> visitation;  // expanded for long continuous sims
  std::vector<float> pheromone;
  std::vector<float> moisture;
  std::vector<uint8_t> biome;  
};

struct SpeciesRecord {
  int species_id{};
  int tick_born{};
  int tick_extinct{-1};  // -1 = still alive
  int peak_population{0};
  float mean_fitness{0.0f};
  std::array<float, 4> centroid_genome{};  
};

struct Metrics {
  int tick{};
  float mean_fitness{};
  float max_fitness{};
  float mean_novelty{};
  float diversity_shannon{};
  int herbivore_count{};
  int predator_count{};
  int species_count{};
  int extinction_events{};
  int speciation_events{};
  
  // Detailed ecology
  float mean_age{};
  float mean_size{};             // Morphology tracking
  float mean_speed{};            // Morphology tracking
  float mean_tox_res{};          // Morphology tracking
  float mean_habitat_match{};
  float mean_resources{};
  float mean_toxicity{};
  
  float total_pheromone{};
  std::array<float, 6> biome_distribution{}; 
  int births{};                  // Spawning events this interval
  int deaths{};                  // Natural + hunted deaths this interval
  WorldEvent active_event{WorldEvent::None};
  float event_intensity{};
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

// Expand density calculation using morphology sensory radius
inline float density_radius(const std::vector<float>& occupancy, int x, int y, float radius, const Config& cfg) {
  float sum = 0.0f;
  int r = static_cast<int>(std::ceil(radius));
  int count = 0;
  for (int dy = -r; dy <= r; ++dy) {
    for (int dx = -r; dx <= r; ++dx) {
      if (dx*dx + dy*dy <= radius*radius) {
        sum += occupancy[idx_2d(x + dx, y + dy, cfg)];
        ++count;
      }
    }
  }
  return count > 0 ? (sum / static_cast<float>(count)) : 0.0f;
}

}  // namespace sim
