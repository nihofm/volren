// Tiny Encryption Algorithm (TEA) to calculate a the seed per launch index and iteration
uint tea(const uint val0, const uint val1, const uint N) {
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;
    for (uint n = 0; n < N; ++n) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
        v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
    }
    return v0;
}

// Return a random sample in the range [0, 1) with a simple Linear Congruential Generator
float rng(inout uint previous) {
    previous = previous * 1664525u + 1013904223u;
    return float(previous & 0X00FFFFFF) / float(0x01000000u);
}
vec2 rng2(inout uint previous) {
    return vec2(rng(previous), rng(previous));
}
vec3 rng3(inout uint previous) {
    return vec3(rng(previous), rng(previous), rng(previous));
}
vec4 rng4(inout uint previous) {
    return vec4(rng(previous), rng(previous), rng(previous), rng(previous));
}
