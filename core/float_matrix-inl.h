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

#ifndef _FLOAT_MATRIX_INL_H_
#define _FLOAT_MATRIX_INL_H_

// IWYU pragma: private, include "float_matrix.h"



inline uint32_t FloatMatrix::get_x_size(void) const {
  return x_size_;
}

inline uint32_t FloatMatrix::get_y_size(void) const {
  return y_size_;
}

// y1x1 y1x2 y1x3 y1x4 y2x1 y2x2 y2x3 y2x4
inline float FloatMatrix::Get(uint32_t x, uint32_t y) const {
  return data_[y * x_size_ + x];
}

inline float &FloatMatrix::Get(uint32_t x, uint32_t y) {
#ifdef _DEBUG
  if (y >= y_size_ || x >= x_size_) {
    fprintf(stderr, "Matrix boundary overrun error");
  }
#endif
  return data_[y * x_size_ + x];
}

inline void FloatMatrix::Set(uint32_t x, uint32_t y, float val) {
#ifdef _DEBUG
  if (y >= y_size_ || x >= x_size_) {
    fprintf(stderr, "Matrix boundary overrun error");
  }
#endif
  data_[y *x_size_ + x] = val;
}

inline uint32_t FloatMatrix::size() const {
  return x_size_ * y_size_;
}

inline const float *FloatMatrix::data(void) const {
  return data_;
}

inline void FloatMatrix::StringWrite(std::string *out) const {
  if (size()) {
    char *source = reinterpret_cast<char*>(data_);
    out->append(source, size() * sizeof(*data_));
  }
}



#endif  // SPEECH_PATTS_ENGINE_COMMON_CORE_FLOAT_MATRIX_INL_H_
