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
/* In-Line FIR filter class */
/*
Author: David Talkin (dtalkin@google.com)
*/

#ifndef _FD_FILTER_H_
#define _FD_FILTER_H_

#include <stdio.h>
#include <stdint.h>

class FFT;

class FdFilter {
 public:
  // Constructor when doing arbitrary FIR filtering using the response
  // in file 'spectrum_shape'
  FdFilter(float input_freq, char* spectrum_shape);
  // Constructor when doing arbitrary FIR filtering using the response
  // in array 'spectrum_array'
  FdFilter(float input_freq, float* spectrum_rray, int n_magnitudes);
  // Constructor when doing highpass or lowpass filtering or when
  // doing rate conversion
  FdFilter(float input_freq, float corner_freq, bool do_highpass,
           float filter_dur, bool do_rate_conversion);

  ~FdFilter();

  // Read and filter/convert all input from inStream and write to
  // outStream until inStream is exhausted.  Return 1 for success 0
  // for failure.
  int FilterStream(FILE* fIn, FILE* fOut);

  // Process nIn samples from 'input' array into 'output' array.  Set
  // 'first' true if this is the first call for a new signal.  Set
  // 'last' true if the end of the signal is in 'input'.  The
  // number of samples transferred to 'output' is the return value.
  // 'maximum_to_output' is the size of caller's output array.
  int FilterArray(int16_t* input, int n_input, bool first, bool last,
                  int16_t* output, int maximum_to_output);

  // Return number of samples left after a call to filterArray() in the
  // case where 'maximum_to_output' is smaller than the number of output samples
  // produced.
  int GetArrayOutputLeftover();

  // Use after a call to filterArray() to get number of 'input'
  // samples actually used (in cases where the caller's output array
  // size has limited the number used, or when the caller's input size
  // is larger than getMaxInputSize()).
  int GetArraySamplesUsed();

  // Return the largest batch of input samples that can be processed
  // with a single call to filterBuffer()
  int GetMaxInputSize();

  // When sample-rate conversion is being done, return the actual
  // output frequency, which may differ from that requested.
  float GetActualOutputFreq();

 private:
  // The main initialization function.  Called by all constructors.
  void FdFilterInitialize(float input_freq, float corner_freq, bool do_highpass,
                          float filter_dur, bool do_rate_conversion,
                          char* spectrum_shape, float* spectrum_array,
                          int n_magnitudes);

  // Use the half filter in fc to create the full filter in filter_coeff_
  void MirrorFilter(float* fc, bool invert);

  // Window approach to creating a high- or low-pass FIR kernel
  void MakeLinearFir(float fc, int* nf, float* coef);

  // Find the closest integer ratio to the float 'a'
  void RationalApproximation(float a, int* k, int* l, int qlim);

  // Complex inner product of length 'n'; result goes to (r3,i3)
  void ComplexDotProduct(int n, float* r1, float* i1, float* r2, float* i2,
                         float* r3, float* i3);

  // Assuming n_input samples are in in_buffer_, process the signal
  // into out_buffer_.  Returns number transferred to out_buffer_ in
  // 'n_output'.
  void FilterBuffer(int n_input, int* n_output);


  int16_t* output_buffer_;  // Processing I/O buffers
  int16_t* input_buffer_;
  float* filter_coeffs_;  // becomes the filter coefficient array
  int n_filter_coeffs_;  // The number of filter coefficients
  int n_filter_coeffs_by2_;  // Precompute for convenience/speed
  int left_over_;  // bookkeeping for the OLA filtering operations
  int first_out_;  // non-zero indicates the first call to the filter kernel
  int to_skip_;  // saves state for the decimator when downsampling
  int array_leftover_;  // Bookkeeping for buffered processing.
  int array_index_; // internal book keeping
  int array_samples_used_; // internal book keeping
  int filter_state_;  // State of processing WRT input and completion
  float true_output_rate_;  // For sample rate conversion, this can
                           // differ from requested freq.
  int16_t* leftovers_;  // saves residual input samples that are mod fft_size
  float* output_delayed_;  // saves samples that exceed the caller's output capacity.
  float* x_;  // FFT processing arrays
  float* y_;
  float* xf_;
  float* yf_;
  int fft_size_;  // Number of points in the FFTs
  int fft_size_by2_;  // Precompute for convenience/speed
  int insert_;  // for sample rate conversion, these are the
                // interpolation/decimations
  int decimate_;
  int max_input_;  // maximum allowed input with each call to filterBuffer
  FFT* fft_;  // The FFT instance used by the filter.
};

#endif  // _FD_FILTER_H_
