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

#ifndef _CODEC_API_H_
#define _CODEC_API_H_

#include <stdint.h>
#include <string>
#include <vector>

#include "wave/codec_riff.h"



class FileResource;

template <class CodecImplementation>
class CodecApi {
 public:
  CodecApi() { }
  ~CodecApi() { }

  // Loads audio samples from a file-resource. It sets the internal header
  // parameters.
  bool Load(FileResource *fr, std::vector<int16_t> *samples);

  // Loads audio samples from a file. It sets the internal header parameters.
  template <class FileResourceType>
  bool Load(const std::string &filename, std::vector<int16_t> *samples);

  // Saves audio samples to a file-resource using the internal header
  // parameters.
  bool Save(const std::vector<int16_t> &samples, FileResource *fr) const;

  // Saves audio samples to a file, using the internal header parameters.
  template <class FileResourceType>
  bool Save(const std::vector<int16_t> &samples, const std::string &filename) const;

  // Sets generic header information.
  bool SetHeaderInfo(int sampling_rate, int num_samples_per_channel);

  // Gets generic header information.
  bool GetHeaderInfo(int *sampling_rate, int *num_samples_per_channel);

  // Reads wave header from file-resource.
  bool ReadHeader(FileResource *fr);

  // Reads the audio data from the FileResource, under the condition that the
  // current position of the FileResource always points to the beginning of the
  // audio-container. The latter condition is preserved by this function. When
  // a codec with internal-state is used (i.e. iSAC), the internal state is kept
  // between sequential reads. The codec is reset prior non-sequential reads.
  // By definition, the codecs DO NOT check whether you are trying to read
  // beyond the boundaries of the audio-data container, so, be careful.
  bool ReadAudioData(int wave_start, int num_samples,
                     std::vector<int16_t> *samples, FileResource *fr);

  // Reads all audio data contained in the audio container held at the current
  // position of the FileResource. The functions assumes and preserves the
  // condition that the current position of the FileResource always points to
  // the beginning of the audio container.
  bool ReadAudioContainer(int container_size_in_bytes,
                          std::vector<int16_t> *samples,
                          FileResource *fr);

  int get_num_samples_per_channel() const;
  int get_sampling_rate() const;

  bool set_num_samples_per_channel(int value);
  bool set_sampling_rate(int value);

  // Initializes the codec for the particular coding type
  bool Initialize(WaveCodingType coding_type);

 private:
  int num_samples_per_channel_;
  int sampling_rate_;
  CodecImplementation codec_;

};


#include "codec_api-inl.h"

#endif  // _CODEC_API_H_
