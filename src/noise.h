#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sim {

inline uint64_t hash2(int x, int y, uint64_t seed) {
  uint64_t h = static_cast<uint64_t>(x) * 0x9e3779b97f4a7c15ULL ^
               static_cast<uint64_t>(y) * 0xbf58476d1ce4e5b9ULL ^ seed;
  h ^= (h >> 30);
  h *= 0xbf58476d1ce4e5b9ULL;
  h ^= (h >> 27);
  h *= 0x94d049bb133111ebULL;
  h ^= (h >> 31);
  return h;
}

inline float grad_dot(uint64_t h, float dx, float dy) {
  constexpr float kTwoPi = 2.0f * 3.14159265358979323846f;
  float angle = static_cast<float>((h & 0xFFFF) / 65535.0) * kTwoPi;
  return std::cos(angle) * dx + std::sin(angle) * dy;
}

inline float fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

inline float perlin2(float x, float y, uint64_t seed) {
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

inline float fbm2(float x, float y, uint64_t seed, int octaves = 7) {
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

}  // namespace sim
