#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <complex>

namespace hybridanc::dsp {

struct AudioBlock {
    float* data = nullptr;
    std::size_t frames = 0;
    std::size_t channels = 1;

    [[nodiscard]] bool valid() const noexcept;
};

struct FIRResult {
    bool valid = false;
};

class FIRFilter {
public:
    explicit FIRFilter(std::size_t tap_count = 1);

    [[nodiscard]] bool set_taps(const float* taps, std::size_t count) noexcept;
    [[nodiscard]] bool process_mono(
        const float* input,
        float* output,
        std::size_t frames
    ) noexcept;
    [[nodiscard]] bool process_stereo(
        const float* input,
        float* output,
        std::size_t frames
    ) noexcept;
    void reset() noexcept;

private:
    std::size_t tap_count_;
    std::vector<float> taps_;
    std::vector<float> history_left_;
    std::vector<float> history_right_;
};

float rms(const float* data, std::size_t count) noexcept;
float peak(const float* data, std::size_t count) noexcept;

struct LimiterConfig {
    float maximum_peak = 0.891250938f;
};

class OutputLimiter {
public:
    explicit OutputLimiter(float maximum_peak = 0.891250938f) noexcept;
    [[nodiscard]] bool process_mono(
        float* data,
        std::size_t frames
    ) noexcept;
    [[nodiscard]] bool process_stereo(
        float* data,
        std::size_t frames
    ) noexcept;
    void reset() noexcept;

private:
    float maximum_peak_;
};

class FFT {
public:
    explicit FFT(std::size_t fft_size);

    [[nodiscard]] bool magnitude(
        const float* input,
        float* output,
        std::size_t frames
    ) noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::size_t fft_size_;
    std::vector<std::complex<float>> scratch_;
};

class LMSFilter {
public:
    LMSFilter(std::size_t filter_length, float learning_rate);

    [[nodiscard]] bool update(
        float reference,
        float desired,
        float& output,
        float& error
    ) noexcept;
    [[nodiscard]] bool process(
        const float* reference,
        const float* desired,
        float* output,
        float* error,
        std::size_t frames
    ) noexcept;

    [[nodiscard]] const std::vector<float>& weights() const noexcept;
    void reset() noexcept;

private:
    std::size_t filter_length_;
    float learning_rate_;
    std::vector<float> weights_;
    std::vector<float> reference_history_;
};

class NLMSFilter {
public:
    NLMSFilter(
        std::size_t filter_length,
        float step_size,
        float epsilon
    );

    [[nodiscard]] bool update(
        float reference,
        float desired,
        float& output,
        float& error
    ) noexcept;
    [[nodiscard]] bool process(
        const float* reference,
        const float* desired,
        float* output,
        float* error,
        std::size_t frames
    ) noexcept;

    [[nodiscard]] const std::vector<float>& weights() const noexcept;
    void reset() noexcept;

private:
    std::size_t filter_length_;
    float step_size_;
    float epsilon_;
    std::vector<float> weights_;
    std::vector<float> reference_history_;
};

class FxLMSFilter {
public:
    FxLMSFilter(
        std::size_t filter_length,
        float learning_rate,
        const float* estimated_secondary,
        std::size_t secondary_length
    );

    [[nodiscard]] bool update(
        float reference,
        float primary_residual,
        float& anti_noise,
        float& residual_error
    ) noexcept;

    [[nodiscard]] bool process(
        const float* reference,
        const float* primary_residual,
        float* anti_noise,
        float* residual_error,
        std::size_t frames
    ) noexcept;

    [[nodiscard]] const std::vector<float>& weights() const noexcept;
    void reset() noexcept;

private:
    std::size_t filter_length_;
    float learning_rate_;
    std::vector<float> weights_;
    std::vector<float> reference_history_;
    std::vector<float> filtered_reference_history_;
    std::vector<float> estimated_secondary_;
};

bool filter_and_limit(
    FIRFilter& fir,
    OutputLimiter& limiter,
    float* data,
    float* scratch,
    std::size_t frames
) noexcept;

} // namespace hybridanc::dsp
