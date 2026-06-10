#pragma once
#include <cstdint>
#include <limits>

using Entity = uint32_t;
constexpr Entity INVALID_ENTITY = std::numeric_limits<Entity>::max();
