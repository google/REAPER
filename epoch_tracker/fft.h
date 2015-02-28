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
// Author: David Talkin (dtalkin@google.com)

#ifndef _FFT_H_
#define _FFT_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.1415927
#endif

#define TRUE 1
#define FALSE 0

class FFT {
 public:
  // Constructor: Prepare for radix-2 FFT's of size (1<<pow2)
  explicit FFT(int pow2);

  ~FFT();

  // Forward fft.  Real time-domain components in x, imaginary in y
  void fft(float *x, float *y);

  // Inverse fft.  Real frequency-domain components in x, imaginary in y
  void ifft(float *x, float *y);

  // Compute the dB-scaled log-magnitude spectrum from the real
  // spectal amplitude values in 'x', and imaginary values in 'y'.
  // Return the magnitude spectrum in z.  Compute 'n' components.
  bool flog_mag(float *x, float *y, float *z, int n);

  // Return the RMS of the spectral density for bands from first_bin
  // to last_bin inclusive.
  float get_band_rms(float *x, float*y, int first_bin, int last_bin);

  // Return the power of 2 required to contain at least size samples.
  static int fft_pow2_from_window_size(int size) {
    int pow2 = 1;
    while ((1 << pow2) < size)
      pow2++;
    return pow2;
  }

  int get_fftSize(void) { return fftSize; }
  int get_power2(void) { return power2; }

 private:
  // Create the trig tables appropriate for transforms of size (1<<pow2).
  int makefttable(int pow2);
  float *fsine, *fcosine;  // The trig tables
  int fft_ftablesize;  // size of trig tables (= (max fft size)/2)
  int power2, kbase, fftSize;  // Misc. values pre-computed for convenience
};

#endif  // _FFT_H_
