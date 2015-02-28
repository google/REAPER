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

/* FdFilter: In-Line Filter class */
/*
Author: David Talkin (dtalkin@google.com)
*/

/*
This class implements a general-purpose streaming FIR filter that is
applied in the frequency domain for speed/efficiency.  Using
FFT-multiply-IFFT rather than a simple time-domain convolution greatly
speeds up the computations.  The longer the impulse response of the
filter, the greater the advantage of this FFT approach over
time-domain convolution.

This class also supports sample-rate conversion.  For simple
input/output rate ratios, it is also quite efficient, but becomes
rather inefficient for ratios with denominator terms greater than 5.
It therefore limits the maximum denominator value to kMaxDecimation,
and thus can only approximate rate conversions that require larger
denominator terms.  For large denominator ratios, or extreme
decimation, indexed time-domain implementations can be much faster.

All filters implemented with FdFilter are symmetric FIR, and thus have
linear phase response.  The filter is implemented non-causally, so
that there is no signal delay introduced by filtering or downsampling
operations.

See the individual methods for details of use.  See fd_filter_msin.cc
for a test harness and use examples.
*/

#include "epoch_tracker/fd_filter.h"

#include <stdio.h>
#include <math.h>

#include "epoch_tracker/fft.h"

// kIoBufferSize can be any reasonable size.
static const int kIoBufferSize = 10000;

// kMaxDecimation sets the limit on the maximum value of the denominator in the
// ratio that approximates the sample rate ratios when rate conversion
// is done.  FdFilter is really not appropriate if the ratio needs to be
// greater than about 7, because time-domain approaches are then likely to be
// faster.  However, perverse or adventuresome souls might want to try
// increasing this if they have extra computer cycles to waste.
static const int kMaxDecimation = 12;

// Constructor when FdFilter is to be used as a high- or low-pass
// filter or as a sample-rate converter.  'input_freq' is the sample
// rate of the input signal in Hz.  If input_freq != corner_freq and
// do_rate_conversion is true, FdFilter is confgured to convert the
// sample rate to corner_freq using a filter of length specified by
// filter_dur.
//
// If do_rate_conversion is false, FdFilter is configured to be a
// high- or low-pass filter.  In this case, corner_freq is interpreted
// as the corner frequency of the filter, and must be in the range 0 <
// corner_freq < (input_freq/2).  If do_highpass is true, the filter
// will be high-pass, else, low-pass.  Again, filter_dur determines
// the filter length.  in all cases, the filter length is specified in
// seconds, and determines the transition band width.  The band width
// is approximately 1.0/filter_dur Hz wide.
/* ******************************************************************** */
FdFilter::FdFilter(float input_freq, float corner_freq, bool do_highpass,
                   float filter_dur, bool do_rate_conversion) {
  FdFilterInitialize(input_freq, corner_freq, do_highpass, filter_dur,
                     do_rate_conversion, NULL, NULL, 0);
}

// Constructor when FdFilter will be used to implement a filter whose
// spectral magnitude values are listed in the plain text file with
// the name specified by 'spectrum_shape'.  input_freq is the sample
// rate in Hz of the input signal.  The file must have the following
// format:
//
// Line 1:                    pow2 nMags
// Lines 2 through (nMags+1): ind val
//
// nMags must equal ((1 << pow2)/2)+1,
// pow2 is typically in the range 3 < pow2 < 15.

// The lines in the file containing (ind, val) pairs specify the
// magnitude response of the filter uniformly sampling the spectrum
// from 0 Hz to (input_freq/2) Hz.  the 'ind' column is just a line index
// to make the file easily human readable, the 'val' column contains
// (positive) magnitude scaling values.  Note that values > 1.0 cause
// an increase in output signal amplitude re the input signal at the
// corresponding frequency, and have the potential to cause clipping,
// if the input signal is too energetic at those frequencies.
/* ******************************************************************** */
FdFilter::FdFilter(float input_freq, char *spectrum_shape) {
  FdFilterInitialize(input_freq, input_freq, 0, 0.01, 0, spectrum_shape,
                     NULL, 0);
}

// Constructor when FdFilter will be used to implement a filter whose
// 'n_magnitudes' spectral magnitude values are in the array 'spectrum_array'.
// n_magnitudes must equal ((1 << pow2)+1) for some pow2 in the
// range 2 < pow2 < 16.  The values in spectrum_array are interpreted as
// above for the 'val' column in the 'spectrum_shape' file.
/* ******************************************************************** */
FdFilter::FdFilter(float input_freq, float *spectrum_array, int n_magnitudes) {
  FdFilterInitialize(input_freq, input_freq, 0, 0.01, 0, NULL,
                     spectrum_array, n_magnitudes);
}

// This is the private method that configures FdFilter to satisfy the
// requirements of the constructor.
/* ----------------------------------------------------------------------- */
void FdFilter::FdFilterInitialize(float input_freq, float corner_freq,
                                  bool do_highpass, float filter_dur,
                                  bool do_rate_conversion,
                                  char *spectrum_shape, float *spectrum_array,
                                  int n_magnitudes) {
  float beta = 0.0, *b = NULL;
  float  ratio_t, ratio, freq1 = input_freq;
  int pow2, i;
  bool do_eq_filtering = false;
  bool b_allocated = true;

  insert_ = 1;
  decimate_ = 1;
  filter_state_ = 1;
  array_leftover_ = 0;
  array_index_ = 0;
  array_samples_used_ = 0;
  true_output_rate_ = input_freq;

  output_buffer_ = new int16_t[kIoBufferSize * 2];
  input_buffer_ = new int16_t[kIoBufferSize];

  if (spectrum_shape || (spectrum_array && (n_magnitudes > 1))) {
    do_eq_filtering = true;
  } else {
    if (do_rate_conversion) {
      freq1 = input_freq;
    } else {   /* it is just a symmetric FIR */
      if (corner_freq >= (freq1 = input_freq) / 2.0) {
        fprintf(stderr,
                "Unreasonable corner frequency specified to filter() (%f)\n",
                corner_freq);
      }
    }
  }

  if (spectrum_shape) {
    FILE *spec_stream = fopen(spectrum_shape, "r");
    if (spec_stream) {
      int n_spect, ind;
      char line[500];
      float fs;
      if (fgets(line, 500, spec_stream)) {
        int nItems = sscanf(line, "%d %d %f", &pow2, &n_spect, &fs);
        if ((nItems == 3) && (fs != input_freq)) {  // This should be a
                                                  // fatal error!
          fprintf(stderr,
                  "Filter spec (%f) does not match input frequency (%f)\n",
                  fs, input_freq);
          fprintf(stderr,
               "The filtering results will probably not be what you want!\n");
        }
        b = new float[n_spect];
        n_filter_coeffs_ = n_spect - 1;  // n_filter_coeffs_ represents actual
                                         // filter-kernel length,
                                         // instead of half filter
                                         // length when using external
                                         // filter.
        for (i = 0; i < n_spect; i++) {
          if ((!fgets(line, 500, spec_stream)) ||
             (sscanf(line, "%d %f", &ind, &(b[i])) != 2)) {
            fprintf(stderr, "Parsing error in spect ratio file %s\n",
                    spectrum_shape);
          }
        }
      } else {
        fprintf(stderr, "Bad format in spectrum file %s\n",
                             spectrum_shape);
      }
      fclose(spec_stream);
    } else {
      fprintf(stderr, "Can't open %s as a spectrum file\n",
                           spectrum_shape);
    }
  } else {
    // Note: n_magnitudes MUST be ((2^k)+1) for k > 1.
    if (spectrum_array && (n_magnitudes > 1)) {
      n_filter_coeffs_ = n_magnitudes - 1;
      int nft = n_filter_coeffs_ * 2;
      pow2 = 1;
      while ((1 << pow2) < nft) {
        pow2++;
      }
      b = spectrum_array;  // Note: b must not be deleted in this case!
      b_allocated = false;
    } else { /* it is not an eq filter */
      if (do_rate_conversion) {
        /* get a ratio of integers close to desired freq. ratio. */
        ratio = corner_freq/freq1;
        RationalApproximation(ratio, &insert_, &decimate_, kMaxDecimation);
        ratio_t = static_cast<float>(insert_) / decimate_;

        if (fabs(1.0 - ratio_t) < .01) {
          fprintf(stderr,
                  "Input and output frequencies are essentially equal!\n");
        }
        true_output_rate_ = ratio_t * freq1;
        // if (corner_freq != true_output_rate_) {
        //   fprintf(stderr,
        // "Warning: Output frequency obtained(%f) is not as requested(%f)\n",
        //    true_output_rate_, corner_freq);
        // }
        corner_freq = true_output_rate_;
        n_filter_coeffs_ = static_cast<int>(freq1 * insert_ * filter_dur) | 1;
        if (corner_freq < freq1)
          beta = (.5 * corner_freq)/(insert_ * freq1);
        else
          beta = .5/insert_;
      } else {
        beta = corner_freq/freq1;
        n_filter_coeffs_ = static_cast<int>(freq1 * filter_dur) | 1;
      }

      /* Generate the symmetric FIR filter coefficients. */
      b = new float[1 + (n_filter_coeffs_ / 2)];
      MakeLinearFir(beta, &n_filter_coeffs_, b);

      if (insert_ > 1) {  // Scale up filter coeffs. to maintain
                        // precision in output.
        float fact = insert_;
        for (i = n_filter_coeffs_ / 2; i >= 0; i--) b[i] *= fact;
      }
    }  // end else it is not an eq filter.
  }  // end else (a spectrum shape was not specified).

  n_filter_coeffs_by2_ = n_filter_coeffs_ / 2;

  if (!do_eq_filtering) { /* Is it a simple high- or low-pass filter? */
    MirrorFilter(b, do_highpass);
    int nf2 = n_filter_coeffs_ << 1;
    fft_size_ = 128;
    pow2 = 7;
    while (nf2 > fft_size_) {
      fft_size_ *= 2;
      pow2++;
    }
  } else {  // It is a filter with the magnitude response specified in b.
    fft_size_ = n_filter_coeffs_ * 2;
    pow2 = 2;
    while ((1 << pow2) < fft_size_)
      pow2++;
  }

  fft_size_by2_ = fft_size_ / 2;

  x_ = new float[fft_size_];
  y_ = new float[fft_size_];
  xf_ = new float[fft_size_];
  yf_ = new float[fft_size_];

  leftovers_ = new int16_t[fft_size_];
  max_input_ = kIoBufferSize / insert_;
  output_delayed_ = new float[(2 * kIoBufferSize)+n_filter_coeffs_+fft_size_];

  float ftscale = 1.0 / fft_size_;
  fft_ = new FFT(pow2);
  if (!do_eq_filtering) {
    // position the filter kernel to be symmetric about time=0
    // Note that this assumes an odd number of symmetric filter coefficients.
    for (i = 0; i <= n_filter_coeffs_by2_; i++) {
      xf_[i] = ftscale * filter_coeffs_[i+n_filter_coeffs_by2_];
      yf_[i] = 0.0;
    }
    for (; i < n_filter_coeffs_; i++) {
      xf_[fft_size_ - n_filter_coeffs_ + i] = ftscale *
           filter_coeffs_[i - n_filter_coeffs_by2_ - 1];
      yf_[fft_size_ - n_filter_coeffs_ + i] = 0.0;
    }
    for (i = n_filter_coeffs_by2_; i < (fft_size_-n_filter_coeffs_by2_); i++)
      xf_[i] = yf_[i] = 0.0;
    fft_->fft(xf_, yf_);
  } else { /* Install the magnitude response symmetrically. */
    for (i = 0; i <= n_filter_coeffs_; i++) {
      xf_[i] = ftscale * b[i];
      yf_[i] = 0.0;
    }
    for (; i < fft_size_; i++) {
      xf_[i] = xf_[fft_size_ - i];
      yf_[i] = 0.0;
    }
  }
  /* The filter, regardless of its origin, is now in the frequency domain. */

  if (b_allocated) {
    delete [] b;
  }
}

// Destructor
/* ******************************************************************** */
FdFilter::~FdFilter() {
  delete [] input_buffer_;
  delete [] output_buffer_;
  delete [] x_;
  delete [] y_;
  delete [] xf_;
  delete [] yf_;
  delete [] filter_coeffs_;
  delete [] leftovers_;
  delete [] output_delayed_;
  delete fft_;
}


// Given the half filter in fc, store the full symmetric kernel in
// filter_coeffs_.
/* ******************************************************************** */
void FdFilter::MirrorFilter(float *fc, bool invert) {
  float *dp1, *dp2, *dp3, sum, integral;
  int i, ncoefb2;

  filter_coeffs_ = new float[n_filter_coeffs_];
  ncoefb2 = 1 + n_filter_coeffs_ / 2;
  // Copy the half-filter and its mirror image into the coefficient array.
  for (i = ncoefb2 - 1, dp3 = fc+ncoefb2 - 1, dp2 = filter_coeffs_,
       dp1 = filter_coeffs_ + n_filter_coeffs_ - 1, integral = 0.0; i-- > 0;) {
    if (!invert) {
      *dp1-- = *dp2++ = *dp3--;
    } else {
      integral += (sum = *dp3--);
      *dp1-- = *dp2++ = -sum;
    }
  }
  if (!invert) {
    *dp1 = *dp3; /* point of symmetry */
  } else {
    integral *= 2;
    integral += *dp3;
    *dp1 = integral - *dp3;
  }
}

// This is a complex vector multiply.  If the second vector (r2, i2) is
// known to be real, set 'i2' to NULL for faster computation.  The
// result is returned in (r3, i3).
/* ******************************************************************** */
void FdFilter::ComplexDotProduct(int n, float *r1, float *i1, float *r2,
                                 float *i2, float *r3, float *i3) {
  float tr1, ti1;

  /* This full complex multiply is only necessary for non-symmetric kernels */
  if (i2) {  // Only supply the i2 vector if you need to do a full
            // complex multiply.
    while (n--) {
      tr1 = (*r1 * *r2) - (*i1 * *i2);
      ti1 = (*r1++ * *i2++) + (*r2++ * *i1++);
      *i3++ = ti1;
      *r3++ = tr1;
    }
  } else {
    /* Can do this iff the filter is symmetric, zero phase.  */
    while (n--) {
      tr1 = (*r1++ * *r2);
      ti1 = (*r2++ * *i1++);
      *i3++ = ti1;
      *r3++ = tr1;
    }
  }
}

/* ******************************************************************** */
// Process samples from 'input' array to 'output' array.  'nIn'
// contains the number of samples available in 'input'.  'maxOut' is
// the maximum number of samples that the caller allows to be
// transferred to 'output' (usually the size of 'output').  Set
// 'first' TRUE if the first sample of a new signal is in 'input'.
// Set 'last' TRUE of the last sample of the signal is in 'input' (to
// cause flushing of processing pipeline).  The simplest setup for use
// of this method requires that the caller use an input buffer no
// larger than the size returned by getMaxInputSize(), and an output
// array twice the size returned by getMaxInputSize().  Then, each
// subsequent call to filterArray will use all 'input' samples, and
// will transfer all available samples to 'output'.  Examples of this
// and the other case, of arbitrarily small caller buffers can be seen
// in the test harness at the end of this file.
// This method returns the number of output samples transferred to 'output'.
int FdFilter::FilterArray(int16_t *input, int nIn, bool first, bool last,
                          int16_t *output, int max_to_output) {
  int i, j, nLeft = nIn, nOut = 0, nToGo = max_to_output;
  int  toRead, toWrite, available;
  int16_t *p = input, *q, *r = output;

  if (first) {
    filter_state_ = 1;  // indicate start of new signal
    array_leftover_ = 0;
    array_index_ = 0;
  }
  if (array_leftover_) {  // First, move any output remaining from the
                         // previous call.
    int toCopy = array_leftover_;
    if (toCopy > nToGo)
      toCopy = nToGo;
    for (i = array_index_, j = 0; j < toCopy; i++, j++)
      *r++ = output_buffer_[i];
    nToGo -= toCopy;
    nOut += toCopy;
    array_leftover_ -= toCopy;
    array_index_ = i;
  }

  if (nToGo <= 0) {
    array_samples_used_ = 0;  // Can't process any input this time; no
                             // room in output array.
    return max_to_output;
  }

  /* Process data from array to array. */
  while (nLeft > 0) {
    toRead = nLeft;
    if (toRead > max_input_)
      toRead = max_input_;
    if (insert_ > 1) {
      for (q = input_buffer_, i = 0; i < toRead; i++) {
        *q++ = *p++;
        for (j = 1; j < insert_; j++)
          *q++ = 0;
      }
    } else {
      for (q = input_buffer_, i = 0; i < toRead; i++)
        *q++ = *p++;
    }
    nLeft -= toRead;
    if ((nLeft <= 0) && last)
      filter_state_ |= 2;  // Indicate that end of signal is (also) in
                           // this bufferful.
    FilterBuffer(toRead * insert_, &available);
    filter_state_ = 0;  // Clear the initialization bit, if any for the
                       // next iteration.

    toWrite = available;
    if (toWrite > nToGo)
      toWrite = nToGo;

    for (i = 0; i < toWrite; i++)
      *r++ = output_buffer_[i];
    nOut += toWrite;
    available -= toWrite;
    if (available > 0) {  // Ran out of output space; suspend processing
      array_leftover_ = available;  // Save the remaining output
                                    // samples for the next call.
      array_index_ = i;
      array_samples_used_ = nIn - nLeft;  // Record the number of input
                                         // samples actually used.
      return(nOut);
    }
  }
  array_samples_used_ = nIn;
  return(nOut);
}

// Use after a call to filterArray() to determine the number of input
// samples actually processed.
int FdFilter::GetArraySamplesUsed() {
  return(array_samples_used_);
}

// Use after a call to filterArray() to determine how many output
// samples were NOT transferred to caller's output array due to lack
// of space in caller's array.
int FdFilter::GetArrayOutputLeftover() {
  return(array_leftover_);
}

// Given the input stream 'input_stream' and the output stream 'output_stream'
// process all samples until EOF is reached on 'input_stream'. Returns 1
// on success, 0 on failure.  All processing is done in a single call
// to this method.
/* ******************************************************************** */
int FdFilter::FilterStream(FILE *input_stream, FILE *output_stream) {
  int i, j;
  int  toread, towrite, nread, rVal = 1, testc;

  toread = max_input_;
  filter_state_ = 1;  // indicate start of new signal
  /* process data from a stream */
  while ((nread = fread(input_buffer_, sizeof(*input_buffer_), toread,
                        input_stream))) {
    testc = getc(input_stream);
    if (feof(input_stream))
      filter_state_ |= 2;
    else
      ungetc(testc, input_stream);
    if (insert_ > 1) {
      int16_t *p, *q;
      for (p = input_buffer_ + nread - 1,
               q = input_buffer_ + (nread * insert_) - 1, i = nread; i--;) {
        for (j = insert_ - 1; j--;) *q-- = 0;
        *q-- = *p--;
      }
    }
    FilterBuffer(nread*insert_, &towrite);
    if ((i = fwrite(output_buffer_, sizeof(*output_buffer_), towrite,
                    output_stream)) < towrite) {
      fprintf(stderr, "Problems writing output in FilterStream\n");
      rVal = 0;
    }
    filter_state_ = 0;
  }
  return(rVal);
}

// This is a private method that supports FilterStream() and
// FilterArray().  It assumes that the input samples have been
// transferred to this->input_buffer_.  It places the filtered results in
// this->output_buffer_, and returns the number of output samples in
// *n_output.
/* ******************************************************************** */
void FdFilter::FilterBuffer(int n_input, int *n_output) {
  int16_t *p, *r, *p2;
  float *dp1, *dp2, *q, half = 0.5;
  int  i, j, k, npass;
  int totaln;

  if (filter_state_ & 1) {  /* first call with this signal and filter? */
    first_out_ = 0;
    to_skip_ = 0;
    left_over_ = 0;
    for (i = max_input_+n_filter_coeffs_, q = output_delayed_; i--;)
      *q++ = 0.0;
  }    /* end of once-per-filter-invocation initialization */

  npass = (n_input + left_over_) / fft_size_;
  if (!npass && (filter_state_ & 2)) {  // if it's the end...
    // Append the input to the leftovers, then pad with zeros.
    p = input_buffer_;
    for (p = input_buffer_, r = leftovers_ + left_over_, i = n_input; i--;) {
      *r++ = *p++; /* append to leftovers_ from prev. call */
    }
    int to_pad = fft_size_ - (left_over_ + n_input);
    for (int i = 0; i < to_pad; ++i) {
      *r++ = 0;
    }
    npass = 1;
  } else {
    /* This is the normal non-boundary course of action. */
    for (p = input_buffer_, r = leftovers_ + left_over_,
           i = fft_size_ - left_over_; i--;)
      *r++ = *p++; /* append to leftovers_ from prev. call */
  }
  if (!npass && !(filter_state_ & 2)) {  // it's not the end, but don't
                                       // have enough data for a loop.
    left_over_ += n_input;
    *n_output = 0;
    first_out_ |= filter_state_ & 1;  // flag that start of sig is still here.
    return;
  }
  filter_state_ |= first_out_;
  first_out_ = 0;

  /* >>>>>>>>>>>>>> Here's the main processing loop. <<<<<<<<<<<<< */
  for (/* p set up above */ q = output_delayed_, i = 0; i < npass;
                            i++, q += fft_size_) {
    if (i) {
      for (r = p + fft_size_by2_, j = fft_size_by2_, dp1 = x_, dp2 = y_; j--;) {
        *dp1++ = *p++;
        *dp2++ = *r++;
      }
      p += fft_size_by2_;
    } else {
      for (p2 = leftovers_, r = p2 + fft_size_by2_, j = fft_size_by2_,
             dp1 = x_, dp2 = y_; j--;) {
        *dp1++ = *p2++;
        *dp2++ = *r++;
      }
    }
    for (j = fft_size_by2_; j--;)
      *dp1++ = *dp2++ = 0.0;

    /* Filtering is done in the frequency domain; transform two real arrays. */
    fft_->fft(x_, y_);
    ComplexDotProduct(fft_size_, x_, y_, xf_, NULL, x_, y_);
    fft_->ifft(x_, y_);

    /* Overlap and add. */
    for (dp2 = q, j = fft_size_ - n_filter_coeffs_by2_; j < fft_size_; j++)
      *dp2++ += x_[j];
    for (j = 0, k = n_filter_coeffs_by2_; j < k; j++)
      *dp2++ += x_[j];
    for (j = n_filter_coeffs_by2_, k = fft_size_ - n_filter_coeffs_by2_;
         j < k; j++)
      *dp2++ = x_[j];

    for (dp2 = q+fft_size_by2_, j = fft_size_ - n_filter_coeffs_by2_;
         j < fft_size_; j++)
      *dp2++ += y_[j];
    for (j = 0, k = n_filter_coeffs_by2_; j < k; j++)
      *dp2++ += y_[j];
    for (j = n_filter_coeffs_by2_, k = fft_size_ - n_filter_coeffs_by2_;
         j < k; j++)
      *dp2++ = y_[j];
  }    /* end of main processing loop */

  left_over_ = n_input - (p-input_buffer_);
  for (i = left_over_, r = leftovers_; i--;)  // Save unused input
                                              // samples for next call.
    *r++ = *p++;
  /* If signal end is here, must process any unused input. */
  if (left_over_ && (filter_state_ & 2)) {  // Must do one more zero-pad pass.
    for (p2 = leftovers_ + left_over_, i = fft_size_ - left_over_; i--;)
      *p2++ = 0;
    for (p2 = leftovers_, r = p2 + fft_size_by2_, j = fft_size_by2_,
           dp1 = x_, dp2 = y_; j--;) {
      *dp1++ = *p2++;
      *dp2++ = *r++;
    }
    for (j = fft_size_by2_; j--;)
      *dp1++ = *dp2++ = 0.0;
    /* Filtering is done in the frequency domain; transform two real arrays. */
    fft_->fft(x_, y_);
    ComplexDotProduct(fft_size_, x_, y_, xf_, NULL, x_, y_);
    fft_->ifft(x_, y_);

    /* Overlap and add. */
    for (dp2 = q, j = fft_size_ - n_filter_coeffs_by2_; j < fft_size_; j++)
      *dp2++ += x_[j];
    for (j = 0, k = n_filter_coeffs_by2_; j < k; j++)
      *dp2++ += x_[j];
    for (j = n_filter_coeffs_by2_, k = fft_size_ - n_filter_coeffs_by2_;
         j < k; j++)
      *dp2++ = x_[j];

    for (dp2 = q+fft_size_by2_, j = fft_size_ - n_filter_coeffs_by2_;
         j < fft_size_; j++)
      *dp2++ += y_[j];
    for (j = 0, k = n_filter_coeffs_by2_; j < k; j++)
      *dp2++ += y_[j];
    for (j = n_filter_coeffs_by2_, k = fft_size_ - n_filter_coeffs_by2_;
         j < k; j++)
      *dp2++ = y_[j];
  }
  /* total good output samples in ob: */
  totaln = (((npass * fft_size_) - to_skip_ -
             ((1 & filter_state_)? n_filter_coeffs_by2_ : 0)) +
            ((filter_state_ & 2)? left_over_ + n_filter_coeffs_by2_ : 0));
  /* number returned to caller: */
  *n_output = 1 + (totaln - 1) / decimate_;  // possible decimation for
                                           // downsampling
  /* Round, decimate and output the samples. */
  float f_temp;
  q = (filter_state_ & 1)? output_delayed_ + (n_filter_coeffs_by2_) :
                                output_delayed_ + to_skip_;
  for (j = decimate_, i =  *n_output, p = output_buffer_; i-- ; q += j) {
    if ((f_temp = *q) > 32767.0)
      f_temp = 32767.0;
    if (f_temp < -32768.0)
      f_temp = -32768.0;
    *p++ = static_cast<int16_t>((f_temp > 0.0) ? half + f_temp : f_temp - half);
  }

  for (dp1 = output_delayed_ + npass*fft_size_, j = n_filter_coeffs_,
         dp2 = output_delayed_; j--;) /*save mem for next call */
    *dp2++ = *dp1++;
  /* If decimating, number to skip on next call. */
  to_skip_ = (*n_output * decimate_) - totaln;
}

// Return the largest number of samples that can be processed in a
// single call to FdFilter::filterArray().  This can be used to configure
// the caller's buffer sizes to simplify subsequent processing.  The
// simplest use of filterArray() is when the caller sends chunks of
// size getMaxInputSize() (or less) as input, and has an output buffer
// of size >= (2 * getMaxInputSize()).  In this case, no checking is
// required to synchronize input and output buffering.  This, and the
// less ideal case of arbitrary caller buffer sizes are illustrated in
// the test harness at the end of this file.
/* ******************************************************************** */
int FdFilter::GetMaxInputSize() {
  return(max_input_);
}

// When sample-rate conversion is attempted, it is possible that the
// output frequency realizable with the FdFilter configuration does not
// exactly match the requested rate.  getActualOutputFreq() retrieves
// the rate achieved, and allows the caller to decide whether to
// proceed with the filtering or not, and to correctly set the rate og
// the output stream for processes later in the chain.  This method
// may be called immediately after instantiation of the FdFilter, or at any
// later time.
float FdFilter::GetActualOutputFreq() {
  return(true_output_rate_);
}

// A private method that finds the closest ratio to the fraction in
// 'a' (0.0 < a < 1.0).  The numerator is returned in 'k', the
// denominator in 'l'.  The largest allowed denominator is specified
// by 'qlim'.
/*      ----------------------------------------------------------      */
void FdFilter::RationalApproximation(float a, int *k, int *l, int qlim) {
  float aa, af, q, em, qq = 1.0, pp = 1.0, ps, e;
  int ai, ip, i;

  aa = fabs(a);
  ai = static_cast<int>(aa);
  i = ai;
  af = aa - i;
  q = 0;
  em = 1.0;
  while (++q <= qlim) {
    ps = q * af;
    ip = static_cast<int>(ps + 0.5);
    e = fabs((ps - static_cast<float>(ip)) / q);
    if (e < em) {
      em = e;
      pp = ip;
      qq = q;
    }
  }
  *k = static_cast<int>((ai * qq) + pp);
  *k = (a > 0)? *k : -(*k);
  *l = static_cast<int>(qq);
}

// A private method to create the coefficients for a symmetric FIR
// lowpass filter using the window technique with a Hanning window.
// Half of the symmetric kernel is returned in ''coef'.  The desired
// number of filter coefficients is in 'nf', but is forced to be odd
// by adding one, if the requesred number is even.  'fc' is the
// normalized corner frequency (0 < fc < 1).
/*      ----------------------------------------------------------      */
void FdFilter::MakeLinearFir(float fc, int *nf, float *coef) {
  int i, n;
  double twopi, fn, c;

  if (((*nf % 2) != 1))
    *nf = *nf + 1;
  n = (*nf + 1) / 2;

  /*  Compute part of the ideal impulse response (the sin(x)/x kernel). */
  twopi = M_PI * 2.0;
  coef[0] = 2.0 * fc;
  c = M_PI;
  fn = twopi * fc;
  for (i = 1; i < n; i++)
    coef[i] = sin(i * fn) / (c * i);

  /* Now apply a Hanning window to the (infinite) impulse response. */
  /* (Could use other windows, like Kaiser, Gaussian...) */
  fn = twopi / *nf;
  for (i = 0; i < n; i++)
    coef[n - i - 1] *= (.5 - (.5 * cos(fn * (i + 0.5))));
}
