#pragma once
#include "noise.h"
#include "types.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace sim {

// ── Biome Classification ─────────────────────────────────────────────────────
inline Biome classify_biome(float height, float temp, float moist) {
  if (height < 0.12f) return Biome::Ocean;
  if (temp < 0.25f) return Biome::Tundra;
  if (moist < 0.25f) return Biome::Desert;
  if (moist > 0.65f && temp > 0.55f) return Biome::Jungle;
  if (moist > 0.40f) return Biome::Forest;
  return Biome::Grassland;
}

// Movement cost multiplier per biome
inline float biome_move_cost(Biome b) {
  switch (b) {
    case Biome::Ocean:     return 99.0f;  // impassable
    case Biome::Tundra:    return 1.4f;
    case Biome::Desert:    return 1.2f;
    case Biome::Grassland: return 1.0f;
    case Biome::Forest:    return 1.3f;
    case Biome::Jungle:    return 1.6f;
    default:               return 1.0f;
  }
}

// Resource carrying capacity multiplier per biome
inline float biome_carrying_capacity(Biome b) {
  switch (b) {
    case Biome::Ocean:     return 0.05f;
    case Biome::Tundra:    return 0.3f;
    case Biome::Desert:    return 0.2f;
    case Biome::Grassland: return 0.8f;
    case Biome::Forest:    return 1.1f;
    case Biome::Jungle:    return 1.4f;
    default:               return 0.6f;
  }
}

// ── World Building ───────────────────────────────────────────────────────────
inline WorldFields build_world(const Config& cfg) {
  WorldFields world;
  const size_t cells = static_cast<size_t>(cfg.width) * static_cast<size_t>(cfg.height);
  world.height.resize(cells);
  world.temperature.resize(cells);
  world.resources.resize(cells);
  world.toxicity.resize(cells, 0.0f);
  world.occupancy.resize(cells, 0.0f);
  world.visitation.resize(cells, 0);
  world.pheromone.resize(cells, 0.0f);
  world.moisture.resize(cells);
  world.biome.resize(cells);

  const float terrain_scale = 0.022f;
  const float climate_scale = 0.009f;
  const float moisture_scale = 0.015f;

  for (int y = 0; y < cfg.height; ++y) {
    for (int x = 0; x < cfg.width; ++x) {
      size_t i = idx_2d(x, y, cfg);

      // Terrain height with island falloff
      float n = fbm2(x * terrain_scale, y * terrain_scale, cfg.seed, 7);
      float cx = (x / static_cast<float>(cfg.width - 1)) - 0.5f;
      float cy = (y / static_cast<float>(cfg.height - 1)) - 0.5f;
      float island = std::max(0.0f, 1.0f - std::sqrt(cx * cx + cy * cy) * 1.6f);
      world.height[i] = (0.52f + 0.48f * n) * island;

      // Temperature: base from climate noise
      float c = fbm2(x * climate_scale, y * climate_scale, cfg.seed + 9999, 5);
      world.temperature[i] = 0.45f + 0.35f * c;

      // Moisture: altitude-dependent (rain shadow) + noise
      float m = fbm2(x * moisture_scale, y * moisture_scale, cfg.seed + 4444, 5);
      float lat = std::abs((y / static_cast<float>(cfg.height - 1)) - 0.5f) * 2.0f;
      float rain_shadow = std::max(0.0f, 1.0f - world.height[i] * 0.8f);
      world.moisture[i] = clamp_value(0.3f + 0.4f * m + 0.2f * rain_shadow - 0.15f * lat, 0.0f, 1.0f);

      // Classify biome
      Biome b = classify_biome(world.height[i], world.temperature[i], world.moisture[i]);
      world.biome[i] = static_cast<uint8_t>(b);

      // Resources modulated by biome carrying capacity
      float base_res = clamp_value(0.6f * island + 0.4f * (0.5f + 0.5f * n), 0.0f, 1.0f);
      world.resources[i] = base_res * biome_carrying_capacity(b);
      // Initialize toxicity (jungles naturally more toxic)
      world.toxicity[i] = (b == Biome::Jungle ? 0.4f : 0.05f) * base_res;
    }
  }
  return world;
}

// ── Climate, Resource, and Pheromone Update ──────────────────────────────────
inline void update_climate_and_resources(WorldFields& world, const Config& cfg, int tick) {
  // Use absolute tick for continuous seasonality
  const float season = std::sin((2.0f * kPi * static_cast<float>(tick)) / 300.0f); // 300 tick seasons
  const ActiveWorldEvent event = current_world_event(cfg, tick);

  const size_t cells = static_cast<size_t>(cfg.width) * static_cast<size_t>(cfg.height);
  std::vector<float> pheromone_next(cells);

  for (int y = 0; y < cfg.height; ++y) {
    for (int x = 0; x < cfg.width; ++x) {
      size_t i = idx_2d(x, y, cfg);
      Biome b = static_cast<Biome>(world.biome[i]);

      // ── Temperature update ──
      const float altitude_cooling = 0.25f * world.height[i];
      const float lat = std::abs((y / static_cast<float>(cfg.height - 1)) - 0.5f) * 2.0f;
      float temp_target = 0.65f - 0.3f * lat - altitude_cooling + 0.14f * season;
      if (event.active) {
        switch (event.type) {
          case WorldEvent::Drought: temp_target += 0.12f * event.intensity; break;
          case WorldEvent::ColdSnap: temp_target -= 0.22f * event.intensity; break;
          case WorldEvent::Bloom: temp_target += 0.03f * event.intensity; break;
          case WorldEvent::ToxicBloom: temp_target += 0.08f * event.intensity; break;
          case WorldEvent::None:
          default: break;
        }
      }
      world.temperature[i] += 0.08f * (temp_target - world.temperature[i]);
      world.temperature[i] = clamp_value(world.temperature[i], 0.0f, 1.0f);

      // ── Moisture update (evaporation in heat, condensation in cold) ──
      float evap = 0.005f * world.temperature[i];
      float condense = 0.003f * (1.0f - world.temperature[i]);
      float moisture_delta = condense - evap + 0.002f * season;
      if (event.active) {
        switch (event.type) {
          case WorldEvent::Drought: moisture_delta -= 0.035f * event.intensity; break;
          case WorldEvent::ColdSnap: moisture_delta += 0.012f * event.intensity; break;
          case WorldEvent::Bloom: moisture_delta += 0.030f * event.intensity; break;
          case WorldEvent::ToxicBloom: moisture_delta += 0.010f * event.intensity; break;
          case WorldEvent::None:
          default: break;
        }
      }
      world.moisture[i] = clamp_value(world.moisture[i] + moisture_delta, 0.0f, 1.0f);

      // Re-classify biome
      world.biome[i] = static_cast<uint8_t>(classify_biome(world.height[i], world.temperature[i], world.moisture[i]));
      b = static_cast<Biome>(world.biome[i]);

      // ── Resource regrowth ──
      const float carrying = biome_carrying_capacity(b);
      float growth = std::max(0.001f, 0.014f * (1.0f - std::abs(world.temperature[i] - 0.55f) * 1.5f));
      if (event.active) {
        switch (event.type) {
          case WorldEvent::Drought: growth *= (1.0f - 0.75f * event.intensity); break;
          case WorldEvent::ColdSnap: growth *= (1.0f - 0.55f * event.intensity); break;
          case WorldEvent::Bloom: growth *= (1.0f + 0.90f * event.intensity); break;
          case WorldEvent::ToxicBloom: growth *= (1.0f - 0.30f * event.intensity); break;
          case WorldEvent::None:
          default: break;
        }
      }
      const float pressure = 0.03f * mean3x3(world.occupancy, x, y, cfg);
      
      float res_increase = growth * world.resources[i] * (carrying - world.resources[i]) - pressure;
      if (event.active) {
        switch (event.type) {
          case WorldEvent::Drought: res_increase -= 0.030f * event.intensity * (0.3f + carrying); break;
          case WorldEvent::ColdSnap: res_increase -= 0.018f * event.intensity; break;
          case WorldEvent::Bloom: res_increase += 0.040f * event.intensity * (0.5f + carrying); break;
          case WorldEvent::ToxicBloom: res_increase -= 0.015f * event.intensity * (0.5f + world.toxicity[i]); break;
          case WorldEvent::None:
          default: break;
        }
      }
      world.resources[i] = clamp_value(world.resources[i] + res_increase, 0.0f, 1.5f);

      // ── Evolving Botany: Toxicity ──
      // Plant toxicity increases with high resource density and grazing pressure.
      float tox_growth = 0.005f * world.resources[i] * (pressure > 0.01f ? 1.5f : 0.5f);
      float tox_decay = 0.002f;
      if (event.active) {
        switch (event.type) {
          case WorldEvent::Drought: tox_growth += 0.006f * event.intensity * world.resources[i]; break;
          case WorldEvent::ColdSnap: tox_decay += 0.001f * event.intensity; break;
          case WorldEvent::Bloom: tox_decay += 0.002f * event.intensity; break;
          case WorldEvent::ToxicBloom:
            tox_growth += 0.012f * event.intensity * (0.4f + world.resources[i]);
            tox_decay *= (1.0f - 0.35f * event.intensity);
            break;
          case WorldEvent::None:
          default: break;
        }
      }
      world.toxicity[i] = clamp_value(world.toxicity[i] + tox_growth - tox_decay, 0.0f, 1.0f);
      
      // ── Pheromone decay + diffusion ──
      float diffused = mean3x3(world.pheromone, x, y, cfg);
      float pheromone_decay = cfg.pheromone_decay;
      if (event.active) {
        switch (event.type) {
          case WorldEvent::Drought: pheromone_decay -= 0.05f * event.intensity; break;
          case WorldEvent::Bloom: pheromone_decay += 0.02f * event.intensity; break;
          case WorldEvent::ColdSnap: pheromone_decay += 0.01f * event.intensity; break;
          case WorldEvent::ToxicBloom: pheromone_decay -= 0.02f * event.intensity; break;
          case WorldEvent::None:
          default: break;
        }
      }
      pheromone_decay = clamp_value(pheromone_decay, 0.75f, 0.98f);
      pheromone_next[i] = pheromone_decay * (0.7f * world.pheromone[i] + 0.3f * diffused);
      pheromone_next[i] = std::max(pheromone_next[i], 0.0f);
      
      // Gradually wipe occupancy
      world.occupancy[i] *= 0.8f; 
    }
  }

  world.pheromone.swap(pheromone_next);
}

}  // namespace sim
