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

#include "wave/codec_riff.h"

#include <limits>

#include "core/file_resource.h"

uint8_t UlawCodec::Int16ToUlaw(int16_t pcm_val) const {
  int16_t mask;
  int16_t seg;
  uint8_t uval;
  // u-law inverts all bits
  // Get the sign and the magnitude of the value.
  if (pcm_val < 0) {
    pcm_val = -pcm_val;
    mask = 0x7f;
  } else {
    mask = 0xff;
  }
  if (pcm_val > kClip) {
    pcm_val = kClip;  // clip the magnitude
  }
  pcm_val += (kBias >> 2);
  // Convert the scaled magnitude to segment number.
  seg = SegmentSearch(pcm_val, seg_uend, 8);
  //  Combine the sign, segment, quantization bits,
  //  and complement the code word.
  if (seg >= 8) {  // out of range, return maximum value.
    return (uint8_t)(0x7f ^ mask);
  } else {
    uval = (uint8_t) (seg << 4) | ((pcm_val >> (seg + 1)) & 0xF);
    return uval ^ mask;
  }
}

int16_t UlawCodec::SegmentSearch(int16_t val, const int16_t *table,
                                 int size) const {
  for (int i = 0; i < size; ++i) {
    if (val <= *table++) {
      return i;
    }
  }
  return size;
}

int16_t UlawCodec::UlawToInt16(uint8_t ulaw) const {
  // Complement to obtain normal u-law value.
  ulaw = ~ulaw;
  //  Extract and bias the quantization bits. Then
  //  shift up by the segment number and subtract out the bias.
  int16_t t = ((ulaw & kQuantMask) << 3) + kBias;
  t <<= ((unsigned)ulaw & kSegMask) >> kSegShift;
  return ((ulaw & kSignBit) ? (kBias - t) : (t - kBias)) >> 2;
}

//
// Table and constants for mu-law coding and decoding:
//
const int16_t UlawCodec::seg_uend[8] = { 0x3f, 0x7f, 0xff, 0x1ff, 0x3ff, 0x7ff,
                                         0xfff, 0x1fff};
const int16_t UlawCodec::kBias = 0x84;      // Bias for linear code.
const int16_t UlawCodec::kClip = 8159;      // max. linear value
const int16_t UlawCodec::kSignBit = 0x80;   // Sign bit for a A-law byte.
const int16_t UlawCodec::kQuantMask = 0xf;  // Quantization field mask.
const int16_t UlawCodec::kSegShift = 4;     // Left shift for segment number.
const int16_t UlawCodec::kSegMask = 0x70;   // Segment field mask.

bool WavRiffCodec::ReadChunk(FileResource *fr, std::string *chunk) const {
  if (chunk == NULL) {
    return false;
  }
  char ch[5];
  if (fread(static_cast<void *>(&ch), 1, 4, fr->fp()) != 4) {
    return false;
  }
  ch[4] = '\0';
  (*chunk) = ch;
  return true;
}

bool WavRiffCodec::CheckChunk(FileResource *fr, const char *data) const {
  std::string chunk;
  if (ReadChunk(fr, &chunk)) {
    return chunk == data;
  }
  return false;
}

bool WavRiffCodec::ReadHeader(FileResource *fr,
                              int32_t *num_samples_per_channel,
                              int32_t *sampling_rate) {
  if (!CheckChunk(fr, "RIFF")) {
    fprintf(stderr, "Invalid file: Expected \"RIFF\" in header");
    return false;
  }
  // File length minus first 8 bytes of RIFF description, we don't use it
  int32_t len = 0;
  if (fread(static_cast<void *>(&len), 4, 1, fr->fp()) != 1) {
    fprintf(stderr, "Failed to read file length");
    return false;
  }

  if (!CheckChunk(fr, "WAVE")) {
    fprintf(stderr, "Invalid file: Expected \"WAVE\" in header");
    return false;
  }

  std::string riff_type;
  if (!ReadChunk(fr, &riff_type)) {
    fprintf(stderr, "Invalid file: Missing RIFF type header");
    return false;
  }

  if (riff_type != "fmt " && riff_type != "bext") {
    fprintf(stderr,
            "Invalid file: RIFF type must be fmt or bext");
    return false;
  }

  // If there is a broadcast format extension, skip it.
  if (riff_type == "bext") {
    int32_t bext_length;
    fread(static_cast<void *>(&bext_length), sizeof(bext_length), 1, fr->fp());
    fseek(fr->fp(), bext_length, SEEK_CUR);
    std::string sub_type;
    if (!CheckChunk(fr, "fmt ")) {
      fprintf(stderr, "Invalid file: fmt subtype not found in bext header");
      return false;
    }
  }

  // Now skip wave format header only reading the number of channels and sample
  // rate:
  if (fread(static_cast<void *>(&len), sizeof(len), 1, fr->fp()) != 1) {
    fprintf(stderr, "failed to skip file length");
    return false;
  }
  if (len < 16) {  // bad format chunk length
    fprintf(stderr, "Invalid of the wave format buffer");
    return false;
  }

  int16_t n_channels = 0;
  fseek(fr->fp(), sizeof(n_channels), SEEK_CUR);
  if (fread(static_cast<void *>(&n_channels), sizeof(n_channels), 1, fr->fp()) != 1) {
    fprintf(stderr, "Failed to read number of channels");
    return false;
  }
  if (n_channels != 1) {
    fprintf(stderr, "Attempt to load multi channel audio");
    return false;
  }
  int32_t sample_rate;
  if (fread(&sample_rate, sizeof(sample_rate), 1, fr->fp()) != 1) {
    fprintf(stderr, "Failed to read sample rate");
    return false;
  }
  fseek(fr->fp(), 8, SEEK_CUR);

  // advance in the stream to skip the wave format block
  fseek(fr->fp(), len - 16, SEEK_CUR);

  // now go to the end of "data" section, if found
  while (!fr->eof()) {
    if (fgetc(fr->fp()) == 'd' &&
        fgetc(fr->fp()) == 'a' &&
        fgetc(fr->fp()) == 't' &&
        fgetc(fr->fp()) == 'a') {
      break;
    }
  }
  if (fr->eof()) {
    fprintf(stderr, "Unexpected end of file: no data");
    return false;
  }

  int32_t num_bytes = 0;
  if (fread(&num_bytes, sizeof(num_bytes), 1, fr->fp()) != 1) {
    fprintf(stderr, "Failed to read number of bytes");
    return false;
  }

  *sampling_rate = sample_rate;
  *num_samples_per_channel = num_bytes / 2;
  return true;
}

bool WavRiffCodec::ReadAudioData(int32_t wave_start,
                                 int32_t num_samples,
                                 std::vector<int16_t> *samples,
                                 FileResource *fr) {
  samples->resize(num_samples);
  bool status = true;
  switch (coding_type_) {
  case PCM16: {
    status &= (fseek(fr->fp(),
                     wave_start * sizeof((*samples)[0]), SEEK_CUR) == 0);
    uint32_t read = fread(&(*samples)[0], sizeof(int16_t), num_samples, fr->fp());
    if (read != num_samples) {
      fprintf(stderr, "WaveIO::ReadSamples: only %d out of %d values read",
          read, num_samples);
      status = false;
    }
  }
  break;
  case ULAW8: {
    UlawCodec sample_codec;
    uint8_t *buffer = new uint8_t[num_samples];
    status &= (fseek(fr->fp(), wave_start * sizeof(*buffer), SEEK_CUR) == 0);
    uint32_t read = fread(buffer, sizeof(uint8_t), num_samples, fr->fp());
    if (read != num_samples) {
      fprintf(stderr, "WaveIO::ReadSamples: only %d out of %d values read",
              read, num_samples);
      status = false;
    }
    for (int i = 0 ; i < num_samples; ++i) {
      (*samples)[i] = sample_codec.UlawToInt16(buffer[i]);
    }
    delete [] buffer;
  }
  break;
  default:
    fprintf(stderr, "WaveIO::ReadSamples: Unsupported coding type (%d)",
            coding_type_);
    status = false;
  }
  return status;
}

bool WavRiffCodec::ReadAudioContainer(int container_size_in_bytes,
                                      std::vector<int16_t> *samples,
                                      FileResource *fr) {
  int number_samples;
  switch (coding_type_) {
  case PCM16:
    number_samples = container_size_in_bytes / 2;
    break;
  case PCM8:
  case ULAW8:
    number_samples = container_size_in_bytes;
    break;
  default:
    return false;
  }
  return ReadAudioData(0, number_samples, samples, fr);
}

bool WavRiffCodec::Initialize(WaveCodingType coding_type) {
  if ((coding_type != PCM16) && (coding_type != PCM8) &&
      (coding_type != ULAW8)) {
    return false;
  }
  coding_type_ = coding_type;
  return true;
}
