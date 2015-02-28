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

#ifndef _WAVE_IO_H_
#define _WAVE_IO_H_

#include "wave/codec_api.h"
#include "wave/codec_riff.h"

class FileResource;

// This class provides the main interface for waveform input/output. The class
// abstracts the implementation of the codec from the rest of the system.
class WaveIO {
 public:
  WaveIO();
  explicit WaveIO(WaveCodingType coding_type);
  ~WaveIO() { }

  template<class FileResourceType>
  bool Load(const std::string &filename,
            std::vector<int16_t> *samples,
            int32_t *sample_rate);

  // Loads the header-info and the audio data from a file, starting from the
  // current position of the FileResource.
  bool Load(FileResource *fr,
            std::vector<int16_t> *samples,
            int32_t *sample_rate);

  // Reads the header-info from the FileResource, starting from the current
  // position.
  bool ReadHeader(FileResource *fr,
                  int32_t *num_samples_per_channel,
                  int32_t *sample_rate);

  // Reads the audio data from the FileResource, under the condition that the
  // current position of the FileResource always points to the beginning of the
  // audio-container. The latter condition is preserved by this function. When
  // a codec with internal-state is used (i.e. iSAC), the internal state is kept
  // between sequential reads. The codec is reset prior non-sequential reads.
  bool ReadAudioData(int32_t wave_start,
                     int32_t num_samples,
                     std::vector<int16_t> *samples,
                     FileResource *fr);

  // Reads all audio data contained in the audio container held at the current
  // position of the FileResource.
  bool ReadAudioContainer(int container_size_in_bytes,
                          std::vector<int16_t> *samples,
                          FileResource *fr);

  // Sets WaveIO to the corresponding codec, and also initializes the codec
  // itself.
  bool Initialize(WaveCodingType coding_type);

  WaveCodingType get_coding_type() const;

 private:
  CodecApi<WavRiffCodec> codec_riff_;
  WaveCodingType coding_type_;
};

#include "wave_io-inl.h"

#endif  // _WAVE_IO_H_
