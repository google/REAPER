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

#ifndef _WAVE_H_
#define _WAVE_H_

#include "core/file_resource.h"
#include <stdint.h>
#include <vector>



class WaveData : public std::vector<int16_t> {
 public:
  WaveData();
  ~WaveData();

  void Set(int number_samples, int sampling_rate, const int16_t *data);
  int sample_rate(void) const;
  void set_sample_rate(int sample_rate);
  bool Equals(const WaveData &wave_data, int threshold = 0) const;

 private:
  int sample_rate_;
};

class Wave {
 public:
  Wave();
  explicit Wave(const WaveData *data);
  ~Wave();

  // Sets internal WaveData to the new object. It loses ownership.
  void set_data(const WaveData *data);

  // Sets values of WaveData to the new data. It takes ownership.
  void copy_data(const WaveData &data);

  // Returns the number of samples contained in the wave.
  int num_samples() const {
    return data_->size();
  }

  // Returns the sample rate of the wave.
  int sample_rate() const {
    return data_->sample_rate();
  }

  // Sets the sample rate for the wave.
  void set_sample_rate(int sample_rate) {
    data_->set_sample_rate(sample_rate);
  }

  // Returns the value of the data at the given offset within the wave.
  int16_t get(int f) const {
    return (*data_)[f];
  }

  // Sets the wave data at the given position with the given value.
  void set(int f, int16_t v) {
    (*data_)[f] = v;
  }

  // Returns the wave data.
  const WaveData *data() const {
    return data_;
  }

  // Ensures that this wave 'parents' the wave data,
  // so deleting it when this instance is deleted.
  void adopt() {
    owner_ = true;
  }

  const int16_t &operator [] (int f) const {
    return (*data_)[f];
  }

  int16_t &operator [] (int f) {
    return (*data_)[f];
  }

  // Resizes
  void resize(int n, bool clear = false);

  // Zero fills the wave data.
  void ZeroFill();

  // Applies the given gain to the wave data, limited to int16_t limits.
  // Returns true if no clipping occurred.
  // Returns false if the amplification resulted in clipping.
  // The gain factor is a simple multiplicative factor, NOT a dB value.
  bool Amplify(float gain_factor);

  // Applies the given gain to the given buffer location and the given
  // number of entries in that buffer.
  // Returns true if no clipping occurred.
  // Returns false if the amplification resulted in clipping.
  // The gain factor is a simple multiplicative factor, NOT a dB value.
  static bool AmplifyBuffer(float gain_factor,
                            int16_t *buffer, uint32_t num_entries);

  bool Load(const std::string &filename);
  bool Load(FileResource *fr);

 protected:
  // Clears the audio data down and sets the owner.
  void Clear();

 private:
  static const int16_t kMaxShort;
  static const int16_t kMinShort;
  // The wave data.
  WaveData *data_;
  // Holds whether this instance is to sole owner of the wave data, allowing
  // for correct object deletion.
  bool owner_;
};

inline bool Wave::Load(const std::string &filename) {
  FileResource fr(filename, "rb");
  if (!fr.Get()) {
    fprintf(stderr, "Failed to open \"%s\"", filename.c_str());
    return false;
  }
  return Load(&fr);
}

#endif  // SPEECH_PATTS_ENGINE_LING_ARCH_WAVE_H_
