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
// A memory mappable float matrix with restricted use, composite audio and
// track classes being its current uses. Its further use should be avoided
// as its designed to meet the specific needs of these classes.

#ifndef _FLOAT_MATRIX_H_
#define _FLOAT_MATRIX_H_

#include <stdint.h>
#include <string>

class FloatMatrix {
 public:
  FloatMatrix();
  FloatMatrix(const FloatMatrix &);
  ~FloatMatrix();

  void operator = (const FloatMatrix &);

  void resize(uint32_t x_size, uint32_t y_size);
  void clear();

  float Get(uint32_t x, uint32_t y) const;
  float &Get(uint32_t x, uint32_t y);
  void Set(uint32_t x, uint32_t y, float val);

  uint32_t get_x_size() const;
  uint32_t get_y_size() const;

  uint32_t size() const;
  const float *data(void) const;

  void StringWrite(std::string *out) const;

 private:
  float *data_;
  uint32_t x_size_;
  uint32_t y_size_;
};

#include "float_matrix-inl.h"

#endif  // _FLOAT_MATRIX_H_
