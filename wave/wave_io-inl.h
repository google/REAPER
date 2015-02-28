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

#ifndef _WAVE_IO_INL_H_
#define _WAVE_IO_INL_H_

#include <string>
#include <vector>

inline WaveCodingType WaveIO::get_coding_type() const {
  return coding_type_;
}

template <class FileResourceType>
bool WaveIO::Load(const std::string &filename,
                  std::vector<int16_t> *samples,
                  int32_t *sample_rate) {
  FileResource fr(filename, "rb");
  if (!fr.Get()) {
    fprintf(stderr, "Failed to open \"%s\"", filename.c_str());
    return false;
  }
  return Load(&fr, samples, sample_rate);
}

#endif  //  _WAVE_IO_INL_H_
