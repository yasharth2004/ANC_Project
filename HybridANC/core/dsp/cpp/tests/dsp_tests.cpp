#include "hybridanc/dsp.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    using namespace hybridanc::dsp;

    float identity[] = {1.0f};
    FIRFilter fir(1);
    assert(fir.set_taps(identity, 1));
    float input[] = {0.25f, -0.5f, 0.75f};
    float output[3] = {};
    assert(fir.process_mono(input, output, 3));
    assert(rms(output, 3) > 0.0f);

    for (float& sample : output) {
        sample = 2.0f;
    }
    OutputLimiter limiter(0.891250938f);
    assert(limiter.process_mono(output, 3));
    assert(peak(output, 3) <= 0.891250938f);

    LMSFilter lms(8, 0.01f);
    float reference[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float desired[] = {0.5f, 0.5f, 0.5f, 0.5f};
    float lms_output[4] = {};
    float lms_error[4] = {};
    assert(lms.process(reference, desired, lms_output, lms_error, 4));
    assert(lms.weights().size() == 8);

    NLMSFilter nlms(8, 0.5f, 1e-6f);
    float nlms_output[4] = {};
    float nlms_error[4] = {};
    assert(nlms.process(reference, desired, nlms_output, nlms_error, 4));
    assert(nlms.weights().size() == 8);

    float secondary[] = {1.0f};
    FxLMSFilter fxlms(8, 0.001f, secondary, 1);
    float anti_noise[4] = {};
    float residual[4] = {};
    assert(fxlms.process(reference, desired, anti_noise, residual, 4));
    assert(fxlms.weights().size() == 8);

    FFT fft(256);
    std::vector<float> sine(256);
    for (std::size_t index = 0; index < sine.size(); ++index) {
        sine[index] = std::sin(
            2.0f * 3.14159265358979323846f
            * 100.0f * static_cast<float>(index) / 48000.0f
        );
    }
    std::vector<float> magnitude(fft.size() / 2 + 1);
    assert(fft.magnitude(sine.data(), magnitude.data(), sine.size()));

    FIRFilter stereo(1);
    float stereo_taps[] = {1.0f};
    assert(stereo.set_taps(stereo_taps, 1));
    float stereo_input[] = {0.1f, -0.2f, 0.3f, 0.4f};
    float stereo_output[4] = {};
    assert(stereo.process_stereo(stereo_input, stereo_output, 2));

    float nan[] = {std::numeric_limits<float>::quiet_NaN()};
    assert(rms(nan, 1) == 0.0f);

    std::cout << "HybridANC C++ DSP tests passed\n";
    return 0;
}
