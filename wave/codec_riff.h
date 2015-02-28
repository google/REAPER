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
limitations under the License.*/

#ifndef _CODEC_RIFF_H_
#define _CODEC_RIFF_H_

#include <stdint.h>
#include <string>
#include <vector>

class FileResource;

enum WaveCodingType {
  PCM16 = 0,
  PCM8,
  ULAW8,
  ISAC_18kbps,
  ISAC_20kbps,
  ISAC_22kbps,
  ISAC_24kbps,
  ISAC_26kbps,
  ISAC_30kbps
};

class WavRiffCodec {
 public:
  WavRiffCodec() { }
  ~WavRiffCodec() { }

  // The header is expected to be in the format:
  //
  // offset    size    description      value
  // 0x00      4       Chunk ID         "RIFF"
  // 0x04      4       Chunk data size  file size - 8
  // 0x08      4       RIFF Type        "WAVE"
  // 0x10              Wave chunks
  //
  // A boolean success indicator can be used to check that the header was read.
  bool ReadHeader(FileResource *fr,
                  int32_t *num_samples_per_channel,
                  int32_t *sample_rate);

  bool ReadAudioData(int32_t wave_start,
                     int32_t num_samples,
                     std::vector<int16_t> *samples,
                     FileResource *fr);

  bool ReadAudioContainer(int container_size_in_bytes,
                          std::vector<int16_t> *samples,
                          FileResource *fr);

  bool Initialize(WaveCodingType coding_type);

 private:
  bool ReadChunk(FileResource *fr, std::string *chunk) const;
  bool CheckChunk(FileResource *fr, const char *data) const;

  WaveCodingType coding_type_;
};

class UlawCodec {
 public:
  uint8_t Int16ToUlaw(int16_t pcm_val) const;
  int16_t UlawToInt16(uint8_t ulaw) const;

 private:
  int16_t SegmentSearch(int16_t val, const int16_t *table, int size) const;

  // Table and constants for mu-law coding and decoding
  static const int16_t seg_uend[8];
  static const int16_t kBias;       // Bias for linear code.
  static const int16_t kClip;       // max. linear value
  static const int16_t kSignBit;    // Sign bit for a A-law byte.
  static const int16_t kQuantMask;  // Quantization field mask.
  static const int16_t kSegShift;   // Left shift for segment number.
  static const int16_t kSegMask;    // Segment field mask.
};

#endif  // _CODEC_RIFF_H_
