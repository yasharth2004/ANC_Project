#include "hybridanc/dsp.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    using namespace hybridanc::dsp;

    constexpr std::size_t frames = 48000;
    constexpr std::size_t runs = 200;

    std::vector<float> reference(frames);
    std::vector<float> desired(frames);
    std::vector<float> output(frames);
    std::vector<float> error(frames);
    for (std::size_t index = 0; index < frames; ++index) {
        reference[index] = std::sin(
            2.0f * 3.14159265358979323846f
            * 100.0f
            * static_cast<float>(index) / 48000.0f
        );
        desired[index] = 0.8f * reference[index];
    }

    LMSFilter lms(64, 0.001f);
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t run = 0; run < runs; ++run) {
        lms.reset();
        const bool valid = lms.process(
            reference.data(),
            desired.data(),
            output.data(),
            error.data(),
            frames
        );
        if (!valid) {
            return 2;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(end - start).count();
    const double seconds_per_second =
        static_cast<double>(runs) / seconds;

    std::cout
        << "C++ LMS: "
        << seconds << " seconds for "
        << runs << " x " << frames
        << " samples, "
        << seconds_per_second << " simulated-seconds/second\n";
    return 0;
}
