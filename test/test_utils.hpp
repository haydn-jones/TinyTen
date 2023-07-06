#pragma once

#include <cstdint>

inline auto rotl(const uint32_t x, int k) -> uint32_t {
    return (x << k) | (x >> (32 - k));
}

static uint32_t s[4] = {0x12345678, 0x87654321, 0xdeadbeef, 0xbadcafe};

auto xoshiro128_p() -> uint32_t {
    const uint32_t result = rotl(s[0] + s[3], 7) + s[0];
    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 11);

    return result;
}
