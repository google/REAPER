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

#ifndef _CODEC_API_INL_H_
#define _CODEC_API_INL_H_

#include "core/file_resource.h"


template <class CodecImplementation>
inline bool CodecApi<CodecImplementation>::SetHeaderInfo(
    int sampling_rate, int num_samples_per_channel) {
  bool status = set_sampling_rate(sampling_rate);
  status &= set_num_samples_per_channel(num_samples_per_channel);
  return status;
}

template <class CodecImplementation>
inline bool CodecApi<CodecImplementation>::GetHeaderInfo(
    int *sampling_rate, int *num_samples_per_channel) {
  *sampling_rate = sampling_rate_;
  *num_samples_per_channel = num_samples_per_channel_;
  return true;
}

template <class CodecImplementation>
inline bool CodecApi<CodecImplementation>::ReadHeader(FileResource *fr) {
  return codec_.ReadHeader(fr, &num_samples_per_channel_, &sampling_rate_);
}

template <class CodecImplementation>
bool CodecApi<CodecImplementation>::ReadAudioData(
    int wave_start, int num_samples,
    std::vector<int16_t> *samples, FileResource *fr) {
  if (samples == NULL) {
    fprintf(stderr, "CodecApi::ReadAudioData: empty pointer was given");
    return true;
  }
  int64_t offset_audio_container = ftell(fr->fp());
  bool status = codec_.ReadAudioData(wave_start, num_samples, samples, fr);
  // Reset the file resource pointer back to the beginning of the
  // audio container.
  if (fseek(fr->fp(), offset_audio_container, SEEK_SET) != 0) {
    fprintf(stderr, "CodecApi::ReadAudioData: error seeking the beginning of the "
            "audio container");
    return false;
  }
  return status;
}

template <class CodecImplementation>
bool CodecApi<CodecImplementation>::ReadAudioContainer(
    int container_size_in_bytes,
    std::vector<int16_t> *samples,
    FileResource *fr) {
  int64_t offset_audio_container = ftell(fr->fp());
  bool status = codec_.ReadAudioContainer(container_size_in_bytes, samples, fr);
  // Reset the FileResource pointer back to the beginning of the audio container
  if (fseek(fr->fp(), offset_audio_container, SEEK_SET) != 0) {
    fprintf(stderr, "CodecApi::ReadAudioData: error seeking the beginning of the "
            "audio container");
    return false;
  }
  return status;
}

template <class CodecImplementation>
inline int CodecApi<CodecImplementation>::get_num_samples_per_channel() const {
  return num_samples_per_channel_;
}

template <class CodecImplementation>
inline int CodecApi<CodecImplementation>::get_sampling_rate() const {
  return sampling_rate_;
}

template <class CodecImplementation>
inline
bool CodecApi<CodecImplementation>::set_num_samples_per_channel(int value) {
  num_samples_per_channel_ = value;
  return true;
}

template <class CodecImplementation>
inline
bool CodecApi<CodecImplementation>::set_sampling_rate(int value) {
  sampling_rate_ = value;
  return true;
}

template <class CodecImplementation>
bool CodecApi<CodecImplementation>::Load(
    FileResource *fr, std::vector<int16_t> *samples) {
  if (!codec_.ReadHeader(fr, &num_samples_per_channel_, &sampling_rate_)) {
    return false;
  }
  samples->resize(num_samples_per_channel_);
  return codec_.ReadAudioData(
      0, num_samples_per_channel_, PCM16, samples, fr);
}

template <class CodecImplementation>
template <class FileResourceType>
bool CodecApi<CodecImplementation>::Load(
    const std::string &filename, std::vector<int16_t> *samples) {
  FileResource *fr = FileResourceType::Open(filename);
  if (fr != NULL) {
    return Load(fr, samples);
  } else {
    fprintf(stderr, "CodecApi::Load: Failed to open \"%s\"", filename.c_str());
    return false;
  }
}

template <class CodecImplementation>
bool CodecApi<CodecImplementation>::Initialize(
    WaveCodingType coding_type) {
  return codec_.Initialize(coding_type);
}


#endif  // SPEECH_PATTS_LIBS_IO_CODEC_API_INL_H_
