#include "hybridanc/dsp.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hybridanc::dsp {

namespace {

bool finite(float value) noexcept {
    return std::isfinite(value);
}

void shift_right(float* data, std::size_t count) noexcept {
    if (count > 1) {
        std::memmove(
            data + 1,
            data,
            (count - 1) * sizeof(float)
        );
    }
}

} // namespace

bool AudioBlock::valid() const noexcept {
    return data != nullptr && frames > 0 && (channels == 1 || channels == 2);
}

FIRFilter::FIRFilter(std::size_t tap_count)
    : tap_count_(tap_count),
      taps_(tap_count, 0.0f),
      history_left_(tap_count, 0.0f),
      history_right_(tap_count, 0.0f) {}

bool FIRFilter::set_taps(
    const float* taps,
    std::size_t count
) noexcept {
    if (taps == nullptr || count == 0 || count != tap_count_) {
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (!finite(taps[index])) {
            return false;
        }
        taps_[index] = taps[index];
    }
    return true;
}

bool FIRFilter::process_mono(
    const float* input,
    float* output,
    std::size_t frames
) noexcept {
    if (input == nullptr || output == nullptr || frames == 0) {
        return false;
    }

    for (std::size_t frame = 0; frame < frames; ++frame) {
        if (!finite(input[frame])) {
            return false;
        }
        shift_right(history_left_.data(), history_left_.size());
        history_left_[0] = input[frame];

        float sum = 0.0f;
        for (std::size_t tap = 0; tap < tap_count_; ++tap) {
            sum += taps_[tap] * history_left_[tap];
            if (!finite(sum)) {
                return false;
            }
        }
        output[frame] = sum;
    }
    return true;
}

bool FIRFilter::process_stereo(
    const float* input,
    float* output,
    std::size_t frames
) noexcept {
    if (input == nullptr || output == nullptr || frames == 0) {
        return false;
    }

    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::size_t channel = 0; channel < 2; ++channel) {
            const std::size_t index = frame * 2 + channel;
            if (!finite(input[index])) {
                return false;
            }
            if (channel == 0) {
                shift_right(history_left_.data(), history_left_.size());
                history_left_[0] = input[index];
            } else {
                shift_right(history_right_.data(), history_right_.size());
                history_right_[0] = input[index];
            }
            float sum = 0.0f;
            const float* history =
                channel == 0
                    ? history_left_.data()
                    : history_right_.data();
            for (std::size_t tap = 0; tap < tap_count_; ++tap) {
                sum += taps_[tap] * history[tap];
                if (!finite(sum)) {
                    return false;
                }
            }
            output[index] = sum;
        }
    }
    return true;
}

void FIRFilter::reset() noexcept {
    std::fill(history_left_.begin(), history_left_.end(), 0.0f);
    std::fill(history_right_.begin(), history_right_.end(), 0.0f);
}

float rms(const float* data, std::size_t count) noexcept {
    if (data == nullptr || count == 0) {
        return 0.0f;
    }
    double square_sum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        if (!finite(data[index])) {
            return 0.0f;
        }
        square_sum +=
            static_cast<double>(data[index])
            * static_cast<double>(data[index]);
    }
    return static_cast<float>(
        std::sqrt(square_sum / static_cast<double>(count))
    );
}

float peak(const float* data, std::size_t count) noexcept {
    if (data == nullptr || count == 0) {
        return 0.0f;
    }
    float result = 0.0f;
    for (std::size_t index = 0; index < count; ++index) {
        if (!finite(data[index])) {
            return 0.0f;
        }
        result = std::max(result, std::abs(data[index]));
    }
    return result;
}

OutputLimiter::OutputLimiter(float maximum_peak) noexcept
    : maximum_peak_(maximum_peak > 0.0f ? maximum_peak : 0.891250938f) {}

bool OutputLimiter::process_mono(
    float* data,
    std::size_t frames
) noexcept {
    if (data == nullptr || frames == 0) {
        return false;
    }
    for (std::size_t index = 0; index < frames; ++index) {
        if (!finite(data[index])) {
            data[index] = 0.0f;
            return false;
        }
        if (std::abs(data[index]) > maximum_peak_) {
            data[index] = std::tanh(
                data[index] / maximum_peak_
            ) * maximum_peak_;
        }
        data[index] = std::clamp(
            data[index],
            -maximum_peak_,
            maximum_peak_
        );
    }
    return true;
}

bool OutputLimiter::process_stereo(
    float* data,
    std::size_t frames
) noexcept {
    if (data == nullptr || frames == 0) {
        return false;
    }
    for (std::size_t index = 0; index < frames * 2; ++index) {
        if (!finite(data[index])) {
            data[index] = 0.0f;
            return false;
        }
        if (std::abs(data[index]) > maximum_peak_) {
            data[index] = std::tanh(
                data[index] / maximum_peak_
            ) * maximum_peak_;
        }
        data[index] = std::clamp(
            data[index],
            -maximum_peak_,
            maximum_peak_
        );
    }
    return true;
}

void OutputLimiter::reset() noexcept {}

FFT::FFT(std::size_t fft_size)
    : fft_size_(fft_size),
      scratch_(fft_size / 2 + 1, std::complex<float>{}) {}

bool FFT::magnitude(
    const float* input,
    float* output,
    std::size_t frames
) noexcept {
    if (input == nullptr || output == nullptr || frames != fft_size_) {
        return false;
    }
    const float pi = 3.14159265358979323846f;
    const std::size_t bins = fft_size_ / 2 + 1;
    for (std::size_t bin = 0; bin < bins; ++bin) {
        std::complex<float> sum{0.0f, 0.0f};
        for (std::size_t sample = 0; sample < fft_size_; ++sample) {
            if (!finite(input[sample])) {
                return false;
            }
            const float angle =
                -2.0f * pi
                * static_cast<float>(bin)
                * static_cast<float>(sample)
                / static_cast<float>(fft_size_);
            sum += std::complex<float>(
                input[sample] * std::cos(angle),
                input[sample] * std::sin(angle)
            );
        }
        scratch_[bin] = sum;
        output[bin] = std::abs(sum);
    }
    return true;
}

std::size_t FFT::size() const noexcept {
    return fft_size_;
}

LMSFilter::LMSFilter(
    std::size_t filter_length,
    float learning_rate
) : filter_length_(filter_length > 0 ? filter_length : 1),
    learning_rate_(learning_rate),
    weights_(filter_length > 0 ? filter_length : 1, 0.0f),
    reference_history_(filter_length > 0 ? filter_length : 1, 0.0f) {}

bool LMSFilter::update(
    float reference,
    float desired,
    float& output,
    float& error
) noexcept {
    if (!finite(reference) || !finite(desired) || !finite(learning_rate_)) {
        return false;
    }
    shift_right(reference_history_.data(), reference_history_.size());
    reference_history_[0] = reference;

    output = 0.0f;
    for (std::size_t index = 0; index < filter_length_; ++index) {
        output += weights_[index] * reference_history_[index];
        if (!finite(output)) {
            return false;
        }
    }
    error = desired - output;
    if (!finite(error)) {
        return false;
    }
    for (std::size_t index = 0; index < filter_length_; ++index) {
        weights_[index] +=
            learning_rate_ * error * reference_history_[index];
        if (!finite(weights_[index])) {
            return false;
        }
    }
    return true;
}

bool LMSFilter::process(
    const float* reference,
    const float* desired,
    float* output,
    float* error,
    std::size_t frames
) noexcept {
    if (reference == nullptr || desired == nullptr
        || output == nullptr || error == nullptr || frames == 0) {
        return false;
    }
    for (std::size_t frame = 0; frame < frames; ++frame) {
        if (!update(
                reference[frame],
                desired[frame],
                output[frame],
                error[frame]
        )) {
            return false;
        }
    }
    return true;
}

const std::vector<float>& LMSFilter::weights() const noexcept {
    return weights_;
}

void LMSFilter::reset() noexcept {
    std::fill(weights_.begin(), weights_.end(), 0.0f);
    std::fill(reference_history_.begin(), reference_history_.end(), 0.0f);
}

NLMSFilter::NLMSFilter(
    std::size_t filter_length,
    float step_size,
    float epsilon
) : filter_length_(filter_length > 0 ? filter_length : 1),
    step_size_(step_size),
    epsilon_(epsilon),
    weights_(filter_length > 0 ? filter_length : 1, 0.0f),
    reference_history_(filter_length > 0 ? filter_length : 1, 0.0f) {}

bool NLMSFilter::update(
    float reference,
    float desired,
    float& output,
    float& error
) noexcept {
    if (!finite(reference) || !finite(desired)
        || !finite(step_size_) || !finite(epsilon_)
        || epsilon_ <= 0.0f) {
        return false;
    }
    shift_right(reference_history_.data(), reference_history_.size());
    reference_history_[0] = reference;

    output = 0.0f;
    for (std::size_t index = 0; index < filter_length_; ++index) {
        output += weights_[index] * reference_history_[index];
    }
    error = desired - output;

    float reference_energy = epsilon_;
    for (float value : reference_history_) {
        reference_energy += value * value;
    }
    for (std::size_t index = 0; index < filter_length_; ++index) {
        weights_[index] +=
            step_size_ * error * reference_history_[index]
            / reference_energy;
        if (!finite(weights_[index])) {
            return false;
        }
    }
    return true;
}

bool NLMSFilter::process(
    const float* reference,
    const float* desired,
    float* output,
    float* error,
    std::size_t frames
) noexcept {
    if (reference == nullptr || desired == nullptr
        || output == nullptr || error == nullptr || frames == 0) {
        return false;
    }
    for (std::size_t frame = 0; frame < frames; ++frame) {
        if (!update(
                reference[frame],
                desired[frame],
                output[frame],
                error[frame]
        )) {
            return false;
        }
    }
    return true;
}

const std::vector<float>& NLMSFilter::weights() const noexcept {
    return weights_;
}

void NLMSFilter::reset() noexcept {
    std::fill(weights_.begin(), weights_.end(), 0.0f);
    std::fill(reference_history_.begin(), reference_history_.end(), 0.0f);
}

FxLMSFilter::FxLMSFilter(
    std::size_t filter_length,
    float learning_rate,
    const float* estimated_secondary,
    std::size_t secondary_length
) : filter_length_(filter_length > 0 ? filter_length : 1),
    learning_rate_(learning_rate),
    weights_(filter_length > 0 ? filter_length : 1, 0.0f),
    reference_history_(filter_length > 0 ? filter_length : 1, 0.0f),
    filtered_reference_history_(filter_length > 0 ? filter_length : 1, 0.0f),
    estimated_secondary_(secondary_length, 0.0f) {
    if (estimated_secondary != nullptr && secondary_length > 0) {
        std::copy(
            estimated_secondary,
            estimated_secondary + secondary_length,
            estimated_secondary_.begin()
        );
    }
}

bool FxLMSFilter::update(
    float reference,
    float primary_residual,
    float& anti_noise,
    float& residual_error
) noexcept {
    if (!finite(reference) || !finite(primary_residual)
        || !finite(learning_rate_)) {
        return false;
    }
    shift_right(reference_history_.data(), reference_history_.size());
    reference_history_[0] = reference;

    anti_noise = 0.0f;
    for (std::size_t index = 0; index < filter_length_; ++index) {
        anti_noise +=
            weights_[index] * reference_history_[index];
    }

    float filtered_reference = 0.0f;
    for (std::size_t tap = 0;
         tap < estimated_secondary_.size();
         ++tap) {
        const std::size_t history_index = tap;
        if (history_index >= reference_history_.size()) {
            break;
        }
        filtered_reference +=
            estimated_secondary_[tap]
            * reference_history_[history_index];
    }

    shift_right(
        filtered_reference_history_.data(),
        filtered_reference_history_.size()
    );
    filtered_reference_history_[0] = filtered_reference;

    residual_error = primary_residual - anti_noise;

    float reference_energy = 1e-12f;
    for (float value : filtered_reference_history_) {
        reference_energy += value * value;
    }
    for (std::size_t index = 0; index < filter_length_; ++index) {
        weights_[index] +=
            learning_rate_
            * residual_error
            * filtered_reference_history_[index]
            / reference_energy;
        if (!finite(weights_[index])) {
            return false;
        }
    }
    return true;
}

bool FxLMSFilter::process(
    const float* reference,
    const float* primary_residual,
    float* anti_noise,
    float* residual_error,
    std::size_t frames
) noexcept {
    if (reference == nullptr || primary_residual == nullptr
        || anti_noise == nullptr || residual_error == nullptr
        || frames == 0) {
        return false;
    }
    for (std::size_t frame = 0; frame < frames; ++frame) {
        if (!update(
                reference[frame],
                primary_residual[frame],
                anti_noise[frame],
                residual_error[frame]
        )) {
            return false;
        }
    }
    return true;
}

const std::vector<float>& FxLMSFilter::weights() const noexcept {
    return weights_;
}

void FxLMSFilter::reset() noexcept {
    std::fill(weights_.begin(), weights_.end(), 0.0f);
    std::fill(reference_history_.begin(), reference_history_.end(), 0.0f);
    std::fill(
        filtered_reference_history_.begin(),
        filtered_reference_history_.end(),
        0.0f
    );
}

bool filter_and_limit(
    FIRFilter& fir,
    OutputLimiter& limiter,
    float* data,
    float* scratch,
    std::size_t frames
) noexcept {
    if (data == nullptr || scratch == nullptr || frames == 0) {
        return false;
    }
    if (!fir.process_mono(data, scratch, frames)) {
        return false;
    }
    std::memcpy(data, scratch, frames * sizeof(float));
    return limiter.process_mono(data, frames);
}

} // namespace hybridanc::dsp
