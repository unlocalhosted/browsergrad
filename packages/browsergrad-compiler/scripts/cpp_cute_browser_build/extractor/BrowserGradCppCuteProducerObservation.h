#pragma once

#include <cstddef>
#include <string_view>

#include "BrowserGradCppCuteProducer.h"

namespace browsergrad::cpp_cute {

bool empty_producer_layout_observation(
    const ProducerLayoutObservation& observation) noexcept;

bool empty_producer_view_copy_observation(
    const ProducerViewCopyObservation& observation) noexcept;

bool producer_view_copy_origins_opened(
    const ProducerPassObservation& pass) noexcept;

std::string_view producer_view_copy_extraction_failure_message(
    const ProducerPassObservation& pass) noexcept;

bool producer_pass_observation_failed(
    const ProducerPassObservation& pass) noexcept;

bool producer_rejected_pass_failed(
    const ProducerReviewResult& producer,
    std::size_t pass_index) noexcept;

}  // namespace browsergrad::cpp_cute
