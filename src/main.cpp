#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kActions = 5;  // up, down, left, right, rest
constexpr float kPi = 3.14159265358979323846f;

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
  std::array<float, 48> genome{};
};

struct Config {
  int width{128};
  int height{128};
  int agents{4096};
  int generations{20};
  int steps_per_generation{220};
  uint64_t seed{7};
  float softmax_temperature{0.8f};
};

struct WorldFields {
  std::vector<float> height;
  std::vector<float> temperature;
  std::vector<float> resources;
  std::vector<float> occupancy;
  std::vector<uint16_t> visitation;
};

uint64_t hash2(int x, int y, uint64_t seed) {
  uint64_t h = static_cast<uint64_t>(x) * 0x9e3779b97f4a7c15ULL ^
               static_cast<uint64_t>(y) * 0xbf58476d1ce4e5b9ULL ^ seed;
  h ^= (h >> 30);
  h *= 0xbf58476d1ce4e5b9ULL;
  h ^= (h >> 27);
  h *= 0x94d049bb133111ebULL;
  h ^= (h >> 31);
  return h;
}

float grad_dot(uint64_t h, float dx, float dy) {
  float angle = static_cast<float>((h & 0xFFFF) / 65535.0) * 2.0f * kPi;
  return std::cos(angle) * dx + std::sin(angle) * dy;
}

float fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

float lerp(float a, float b, float t) { return a + (b - a) * t; }

float perlin2(float x, float y, uint64_t seed) {
  int x0 = static_cast<int>(std::floor(x));
  int y0 = static_cast<int>(std::floor(y));
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float sx = x - static_cast<float>(x0);
  float sy = y - static_cast<float>(y0);

  float n00 = grad_dot(hash2(x0, y0, seed), sx, sy);
  float n10 = grad_dot(hash2(x1, y0, seed), sx - 1.0f, sy);
  float n01 = grad_dot(hash2(x0, y1, seed), sx, sy - 1.0f);
  float n11 = grad_dot(hash2(x1, y1, seed), sx - 1.0f, sy - 1.0f);

  float u = fade(sx);
  float v = fade(sy);
  return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v);
}

float fbm2(float x, float y, uint64_t seed, int octaves = 7) {
  float sum = 0.0f;
  float amp = 1.0f;
  float freq = 1.0f;
  float norm = 0.0f;
  for (int i = 0; i < octaves; ++i) {
    sum += amp * perlin2(x * freq, y * freq, seed + static_cast<uint64_t>(i) * 101ULL);
    norm += amp;
    amp *= 0.5f;
    freq *= 2.0f;
  }
  return sum / std::max(norm, 1e-6f);
}

size_t idx_2d(int x, int y, const Config& cfg) {
  x = std::clamp(x, 0, cfg.width - 1);
  y = std::clamp(y, 0, cfg.height - 1);
  return static_cast<size_t>(y) * static_cast<size_t>(cfg.width) + static_cast<size_t>(x);
}

WorldFields build_world(const Config& cfg) {
  WorldFields world;
  const size_t cells = static_cast<size_t>(cfg.width) * static_cast<size_t>(cfg.height);
  world.height.resize(cells);
  world.temperature.resize(cells);
  world.resources.resize(cells);
  world.occupancy.resize(cells);
  world.visitation.resize(cells);

  const float terrain_scale = 0.022f;
  const float climate_scale = 0.009f;
  for (int y = 0; y < cfg.height; ++y) {
    for (int x = 0; x < cfg.width; ++x) {
      size_t i = idx_2d(x, y, cfg);
      float n = fbm2(x * terrain_scale, y * terrain_scale, cfg.seed, 7);
      float cx = (x / static_cast<float>(cfg.width - 1)) - 0.5f;
      float cy = (y / static_cast<float>(cfg.height - 1)) - 0.5f;
      float island = std::max(0.0f, 1.0f - std::sqrt(cx * cx + cy * cy) * 1.6f);
      world.height[i] = (0.52f + 0.48f * n) * island;

      float c = fbm2(x * climate_scale, y * climate_scale, cfg.seed + 9999, 5);
      world.temperature[i] = 0.45f + 0.35f * c;
      world.resources[i] = std::clamp(0.6f * island + 0.4f * (0.5f + 0.5f * n), 0.0f, 1.0f);
      world.occupancy[i] = 0.0f;
      world.visitation[i] = 0;
    }
  }
  return world;
}

float mean3x3(const std::vector<float>& field, int x, int y, const Config& cfg) {
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

void reset_dynamic_fields(WorldFields& world) {
  std::fill(world.occupancy.begin(), world.occupancy.end(), 0.0f);
  std::fill(world.visitation.begin(), world.visitation.end(), 0);
}

int classify_species(const Agent& a) {
  // Cheap locality-sensitive species hash from first genes.
  const int g0 = static_cast<int>(std::floor((a.genome[0] + 2.0f) * 4.0f));
  const int g1 = static_cast<int>(std::floor((a.genome[1] + 2.0f) * 4.0f));
  const int g2 = static_cast<int>(std::floor((a.genome[2] + 2.0f) * 4.0f));
  return (g0 * 17 + g1 * 7 + g2 * 3) & 63;
}

void init_population(std::vector<Agent>& population, const Config& cfg, std::mt19937_64& rng) {
  std::uniform_real_distribution<float> xdist(0.0f, static_cast<float>(cfg.width - 1));
  std::uniform_real_distribution<float> ydist(0.0f, static_cast<float>(cfg.height - 1));
  std::normal_distribution<float> gdist(0.0f, 0.45f);

  population.resize(static_cast<size_t>(cfg.agents));
  for (auto& a : population) {
    a.pos = {xdist(rng), ydist(rng)};
    a.energy = 6.0f;
    a.fitness = 0.0f;
    a.novelty_score = 0.0f;
    a.age = 0;
    for (float& g : a.genome) {
      g = gdist(rng);
    }
    a.species_id = classify_species(a);
  }
}

std::array<float, kActions> policy_logits(const Agent& a, const Config& cfg,
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

  std::array<float, 8> feat{nx, ny, h, t, r, density, e, season_phase};
  std::array<float, kActions> logits{};
  for (int action = 0; action < kActions; ++action) {
    float s = a.genome[static_cast<size_t>(action) * feat.size() + (feat.size() - 1)];
    for (size_t k = 0; k < feat.size(); ++k) {
      s += a.genome[static_cast<size_t>(action) * feat.size() + k] * feat[k];
    }
    logits[action] = s;
  }
  return logits;
}

int sample_action(const std::array<float, kActions>& logits, float temperature,
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

void update_climate_and_resources(WorldFields& world, const Config& cfg, int step) {
  const float season = std::sin((2.0f * kPi * static_cast<float>(step)) /
                                std::max(30.0f, static_cast<float>(cfg.steps_per_generation)));

  for (int y = 0; y < cfg.height; ++y) {
    for (int x = 0; x < cfg.width; ++x) {
      size_t i = idx_2d(x, y, cfg);
      const float altitude_cooling = 0.25f * world.height[i];
      const float lat = std::abs((y / static_cast<float>(cfg.height - 1)) - 0.5f) * 2.0f;

      float temp_target = 0.65f - 0.3f * lat - altitude_cooling + 0.14f * season;
      world.temperature[i] += 0.08f * (temp_target - world.temperature[i]);
      world.temperature[i] = std::clamp(world.temperature[i], 0.0f, 1.0f);

      // Resource regrowth: logistic growth modulated by temperature and occupancy pressure.
      const float growth = 0.014f * (1.0f - std::abs(world.temperature[i] - 0.55f) * 1.5f);
      const float carrying = std::clamp(0.2f + 0.9f * (1.0f - world.height[i] * 0.35f), 0.1f, 1.1f);
      const float pressure = 0.03f * mean3x3(world.occupancy, x, y, cfg);
      world.resources[i] += growth * world.resources[i] * (carrying - world.resources[i]) - pressure;
      world.resources[i] = std::clamp(world.resources[i], 0.0f, 1.2f);
    }
  }
}

void step_agents(std::vector<Agent>& population, const Config& cfg, WorldFields& world,
                 int step, std::mt19937_64& rng) {
  const float season_phase = std::sin((2.0f * kPi * static_cast<float>(step)) /
                                      std::max(40.0f, static_cast<float>(cfg.steps_per_generation)));

  for (auto& a : population) {
    int x = static_cast<int>(a.pos.x);
    int y = static_cast<int>(a.pos.y);

    int action = sample_action(policy_logits(a, cfg, world, x, y, season_phase),
                               cfg.softmax_temperature, rng);

    float dx = 0.0f;
    float dy = 0.0f;
    if (action == 0) dy = -1.0f;
    if (action == 1) dy = 1.0f;
    if (action == 2) dx = -1.0f;
    if (action == 3) dx = 1.0f;

    a.pos.x = std::clamp(a.pos.x + dx, 0.0f, static_cast<float>(cfg.width - 1));
    a.pos.y = std::clamp(a.pos.y + dy, 0.0f, static_cast<float>(cfg.height - 1));
    x = static_cast<int>(a.pos.x);
    y = static_cast<int>(a.pos.y);

    const size_t i = idx_2d(x, y, cfg);
    world.occupancy[i] += 1.0f;
    world.visitation[i] = static_cast<uint16_t>(std::min(65535, world.visitation[i] + 1));

    const float move_cost = 0.025f + 0.018f * (std::abs(dx) + std::abs(dy));
    const float thermal_penalty = 0.04f * std::abs(world.temperature[i] - 0.58f);

    // Resource harvest with depletion.
    const float harvest = std::min(world.resources[i], 0.11f + 0.015f * (action == 4 ? 1.0f : 0.0f));
    world.resources[i] -= harvest;

    // Intrinsic novelty reward to promote exploration.
    const float novelty = 1.0f / std::sqrt(1.0f + static_cast<float>(world.visitation[i]));
    a.novelty_score += novelty;

    // Social reward from moderate density (coordination without overcrowding).
    const float density = mean3x3(world.occupancy, x, y, cfg);
    const float social = std::exp(-std::pow(density - 2.2f, 2.0f) / 3.0f) * 0.03f;

    const float reward = 0.9f * harvest + 0.06f * novelty + social - move_cost - thermal_penalty;
    a.energy += reward;
    a.fitness += reward;

    a.age += 1;
    if (a.energy < 0.0f) {
      a.fitness -= 1.0f;
      a.energy = 0.0f;
    }
  }
}

void evolve(std::vector<Agent>& population, const Config& cfg, std::mt19937_64& rng) {
  std::sort(population.begin(), population.end(), [](const Agent& a, const Agent& b) {
    return (a.fitness + 0.1f * a.novelty_score) > (b.fitness + 0.1f * b.novelty_score);
  });

  const int elite_count = std::max(4, cfg.agents / 12);
  std::vector<Agent> elites(population.begin(), population.begin() + elite_count);

  std::uniform_int_distribution<int> pick(0, elite_count - 1);
  std::uniform_real_distribution<float> coin(0.0f, 1.0f);
  std::normal_distribution<float> mut(0.0f, 0.035f);
  std::uniform_real_distribution<float> xdist(0.0f, static_cast<float>(cfg.width - 1));
  std::uniform_real_distribution<float> ydist(0.0f, static_cast<float>(cfg.height - 1));

  std::vector<Agent> next(static_cast<size_t>(cfg.agents));
  for (auto& child : next) {
    const Agent& p1 = elites[pick(rng)];
    const Agent& p2 = elites[pick(rng)];

    for (size_t i = 0; i < child.genome.size(); ++i) {
      float inherited = (coin(rng) < 0.5f) ? p1.genome[i] : p2.genome[i];
      // Adaptive mutation: higher if parents underperform.
      float adapt = 1.0f + std::max(0.0f, 0.5f - 0.5f * (p1.fitness + p2.fitness) / 100.0f);
      if (coin(rng) < 0.18f) inherited += mut(rng) * adapt;
      child.genome[i] = inherited;
    }

    child.pos = {xdist(rng), ydist(rng)};
    child.energy = 6.0f;
    child.fitness = 0.0f;
    child.novelty_score = 0.0f;
    child.age = 0;
    child.species_id = classify_species(child);
  }

  population.swap(next);
}

struct Metrics {
  float mean_fitness{};
  float best_fitness{};
  float mean_novelty{};
  float diversity_shannon{};
};

Metrics compute_metrics(const std::vector<Agent>& population) {
  Metrics m;
  if (population.empty()) return m;

  std::unordered_map<int, int> species_counts;
  species_counts.reserve(128);

  float sum_f = 0.0f;
  float sum_n = 0.0f;
  float best = -1e9f;
  for (const auto& a : population) {
    sum_f += a.fitness;
    sum_n += a.novelty_score;
    best = std::max(best, a.fitness);
    species_counts[a.species_id] += 1;
  }

  m.mean_fitness = sum_f / static_cast<float>(population.size());
  m.mean_novelty = sum_n / static_cast<float>(population.size());
  m.best_fitness = best;

  float h = 0.0f;
  const float n = static_cast<float>(population.size());
  for (const auto& kv : species_counts) {
    float p = kv.second / n;
    if (p > 0.0f) h -= p * std::log(std::max(p, 1e-8f));
  }
  m.diversity_shannon = h;
  return m;
}

std::string summary_json(const Config& cfg, const std::vector<Metrics>& metrics) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(4);
  os << "{\n";
  os << "  \"seed\": " << cfg.seed << ",\n";
  os << "  \"world\": {\"width\": " << cfg.width << ", \"height\": " << cfg.height << "},\n";
  os << "  \"agents\": " << cfg.agents << ",\n";
  os << "  \"config\": {\"steps_per_generation\": " << cfg.steps_per_generation
     << ", \"softmax_temperature\": " << cfg.softmax_temperature << "},\n";
  os << "  \"generations\": [\n";

  for (size_t i = 0; i < metrics.size(); ++i) {
    os << "    {\"generation\": " << i
       << ", \"mean_fitness\": " << metrics[i].mean_fitness
       << ", \"best_fitness\": " << metrics[i].best_fitness
       << ", \"mean_novelty\": " << metrics[i].mean_novelty
       << ", \"species_shannon\": " << metrics[i].diversity_shannon << "}";
    if (i + 1 != metrics.size()) os << ",";
    os << "\n";
  }

  os << "  ]\n";
  os << "}\n";
  return os.str();
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  if (argc > 1) cfg.agents = std::max(100, std::stoi(argv[1]));
  if (argc > 2) cfg.generations = std::max(1, std::stoi(argv[2]));
  if (argc > 3) cfg.steps_per_generation = std::max(1, std::stoi(argv[3]));
  if (argc > 4) cfg.softmax_temperature = std::max(0.05f, std::stof(argv[4]));

  std::mt19937_64 rng(cfg.seed);
  WorldFields world = build_world(cfg);

  std::vector<Agent> population;
  init_population(population, cfg, rng);

  std::vector<Metrics> metrics;
  metrics.reserve(static_cast<size_t>(cfg.generations));

  for (int gen = 0; gen < cfg.generations; ++gen) {
    for (auto& a : population) {
      a.energy = 6.0f;
      a.fitness = 0.0f;
      a.novelty_score = 0.0f;
      a.age = 0;
      a.species_id = classify_species(a);
    }

    reset_dynamic_fields(world);
    for (int step = 0; step < cfg.steps_per_generation; ++step) {
      update_climate_and_resources(world, cfg, step + gen * cfg.steps_per_generation);
      step_agents(population, cfg, world, step, rng);
    }

    const Metrics m = compute_metrics(population);
    metrics.push_back(m);

    std::cout << "gen=" << gen << " mean=" << std::fixed << std::setprecision(4)
              << m.mean_fitness << " best=" << m.best_fitness
              << " novelty=" << m.mean_novelty
              << " species_H=" << m.diversity_shannon << "\n";

    if (gen + 1 < cfg.generations) {
      evolve(population, cfg, rng);
    }
  }

  std::ofstream out("simulation_summary.json", std::ios::binary);
  out << summary_json(cfg, metrics);
  out.close();
  std::cout << "Wrote simulation_summary.json\n";
  return 0;
}
