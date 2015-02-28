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

#include "core/float_matrix.h"

#include <string.h>



FloatMatrix::FloatMatrix() {
  data_ = NULL;
  x_size_ = 0;
  y_size_ = 0;
}

FloatMatrix::FloatMatrix(const FloatMatrix &m) {
  clear();
  resize(m.x_size_, m.y_size_);
  for (uint32_t i = 0; i < x_size_ * y_size_; i++) {
    data_[i] = m.data_[i];
  }
}

FloatMatrix::~FloatMatrix() {
  clear();
}

void FloatMatrix::operator=(const FloatMatrix &m) {
  clear();
  resize(m.x_size_, m.y_size_);
  for (uint32_t i = 0; i < x_size_ * y_size_; i++) {
    data_[i] = m.data_[i];
  }
}

void FloatMatrix::clear() {
  if (data_ != NULL) {
    delete [] data_;
    data_ = NULL;
  }
  x_size_ = 0;
  y_size_ = 0;
}

void FloatMatrix::resize(uint32_t x_size, uint32_t y_size) {
  if (data_ == NULL) {
    data_ = new float [x_size * y_size];
    memset(data_, 0, sizeof(float)*x_size * y_size);
  } else if (x_size == x_size_ && y_size == y_size_) {
    return;
  } else {
    float *newdata = new float [x_size * y_size];
    memset(newdata, 0, sizeof(float) * x_size * y_size);
    int min_x = x_size < x_size_ ? x_size : x_size_;
    int min_y = y_size < y_size_ ? y_size : y_size_;
    for (int y = 0; y < min_y; y++) {
      memcpy(newdata + y * x_size, data_ + y * x_size_, min_x * sizeof(float));
    }
    delete [] data_;
    data_ = newdata;
  }

  x_size_ = x_size;
  y_size_ = y_size;
}
