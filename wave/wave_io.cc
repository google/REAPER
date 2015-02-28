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

#include "wave/wave_io.h"

#include <limits>

#include "core/file_resource.h"



WaveIO::WaveIO(void) {
  Initialize(PCM16);
}

WaveIO::WaveIO(WaveCodingType coding_type) {
  Initialize(coding_type);
}

bool WaveIO::ReadAudioData(int32_t wave_start,
                           int32_t num_samples,
                           std::vector<int16_t> *samples,
                           FileResource *fr) {
  switch (coding_type_) {
    case PCM16:
    case PCM8:
    case ULAW8:
      return codec_riff_.ReadAudioData(wave_start, num_samples, samples, fr);
    default:
      fprintf(stderr, "WaveIO:ReadAudioData: unknown coding type %d",
              coding_type_);
  }
  return false;
}

bool WaveIO::ReadAudioContainer(int container_size_in_bytes,
                                std::vector<int16_t> *samples,
                                FileResource *fr) {
  switch (coding_type_) {
    case PCM16:
    case PCM8:
    case ULAW8:
      return codec_riff_.ReadAudioContainer(container_size_in_bytes,
                                            samples, fr);
      fprintf(stderr, "WaveIO:ReadAudioData: unknown coding type %d",
              coding_type_);
    default:
      fprintf(stderr, "WaveIO: coding type not supported %d", coding_type_);
      return false;
  }
  return false;
}

bool WaveIO::ReadHeader(FileResource *fr,
                        int32_t *num_samples_per_channel,
                        int32_t *sampling_rate) {
  switch (coding_type_) {
    case PCM16:
    case PCM8:
    case ULAW8: {
      bool status = codec_riff_.ReadHeader(fr);
      if (!status) {
        return false;
      }
      *num_samples_per_channel = codec_riff_.get_num_samples_per_channel();
      *sampling_rate = codec_riff_.get_sampling_rate();
      return true;
    }
    default:
      fprintf(stderr, "WaveIO:ReadHeader: unknown coding type %d",
              coding_type_);
  }
  return false;
}

bool WaveIO::Load(FileResource *fr,
                  std::vector<int16_t> *samples,
                  int32_t *sample_rate) {
  int32_t num_samples;
  if (!ReadHeader(fr, &num_samples, sample_rate)) {
    return false;
  }
  samples->resize(num_samples);
  return ReadAudioData(0, num_samples, samples, fr);
}

bool WaveIO::Initialize(WaveCodingType coding_type) {
  coding_type_ = coding_type;
  switch (coding_type_) {
    case PCM16:
    case PCM8:
    case ULAW8:
      return codec_riff_.Initialize(coding_type);
    default:
      fprintf(stderr, "WaveIO:WaveIO: unknown coding type %d", coding_type_);
      return false;
  }
}
