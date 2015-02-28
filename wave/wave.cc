/*
Copyright 2015 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "wave/wave.h"

#include <limits>
#include <stdlib.h>
#include <vector>

#include "core/file_resource.h"
#include "wave/wave_io.h"



WaveData::WaveData() : sample_rate_(16000) {
}
WaveData::~WaveData() {
}

void WaveData::Set(int number_samples, int sampling_rate, const int16_t *data) {
  set_sample_rate(sampling_rate);
  resize(number_samples);
  for (int n = 0; n < number_samples; ++n) {
    (*this)[n] = data[n];
  }
}

int WaveData::sample_rate(void) const {
  return sample_rate_;
}

void WaveData::set_sample_rate(int sample_rate) {
  sample_rate_ = sample_rate;
}

bool WaveData::Equals(const WaveData &wave_data, int threshold) const {
  if (wave_data.size() != size()) {
    fprintf(stderr, "Different number of samples");
    return false;
  }
  if (wave_data.sample_rate() != sample_rate()) {
    fprintf(stderr, "Different sample rate");
    return false;
  }
  for (int i = 0; i < size(); ++i) {
    if (abs(wave_data[i] - (*this)[i]) > threshold) {
      fprintf(stderr, "Sample %d differs", i);
      return false;
    }
  }
  return true;
}

Wave::Wave() {
  data_ = new WaveData;
  owner_ = true;
}

Wave::Wave(const WaveData *data) {
  // Shallow copy
  data_ = const_cast<WaveData *>(data);
  owner_ = false;
}

Wave::~Wave() {
  Clear();
}

void Wave::Clear() {
  if (owner_) {
    delete data_;
  }
  data_ = NULL;
  owner_ = true;
}

void Wave::set_data(const WaveData *data) {
  if (owner_) {
    delete data_;
  }
  data_ = const_cast<WaveData *>(data);
  owner_ = false;
}

void Wave::copy_data(const WaveData &data) {
  if (owner_) {
    delete data_;
  }
  data_ = new WaveData(data);
  owner_ = true;
}

void Wave::resize(int n, bool clear) {
  if (!owner_) {
    // copy the data to a local copy
    WaveData *new_wave_data = new WaveData(*data_);
    new_wave_data->set_sample_rate(data_->sample_rate());
    data_ = new_wave_data;
  }
  data_->resize(n);
  owner_ = true;
}

void Wave::ZeroFill() {
  if (!owner_) {
    // copy the data to a local copy
    WaveData *new_wave_data = new WaveData(*data_);
    new_wave_data->set_sample_rate(data_->sample_rate());
    data_ = new_wave_data;
    owner_ = true;
  }
  std::fill(data_->begin(), data_->end(), 0);
}

bool Wave::Load(FileResource *fr) {
  WaveIO wave_io;
  std::vector<int16_t> *samples = reinterpret_cast<std::vector<int16_t> *>(data_);
  int sampling_rate;
  bool status = wave_io.Load(fr, samples, &sampling_rate);
  set_sample_rate(sampling_rate);
  return status;
}

const int16_t Wave::kMaxShort = std::numeric_limits<int16_t>::max() - 1;
const int16_t Wave::kMinShort = std::numeric_limits<int16_t>::min() + 1;

bool Wave::Amplify(float gain) {
  return AmplifyBuffer(gain, &(*data_)[0], data_->size());
}

bool Wave::AmplifyBuffer(float gain, int16_t *buf, uint32_t size) {
  bool clipped = false;
  for (uint32_t i = 0; i < size; ++i) {
    float sample = static_cast<float>(buf[i]) * gain;
    if (sample > kMaxShort) {
      sample = kMaxShort;
      clipped = true;
    } else if (sample < kMinShort) {
      sample = kMinShort;
      clipped = true;
    }
    buf[i] = static_cast<int16_t>(sample);
  }
  return !clipped;
}
