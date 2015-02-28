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

// LpcAnalyzer
// A collection of methods commonly used in linear-prediction analysis.

#ifndef _LPC_ANALYZER_H_
#define _LPC_ANALYZER_H_

#include <vector>

// Largest order allowed for any linear predictor analysis
#define BIGSORD 100

class LpcAnalyzer {
 public:
  LpcAnalyzer(void) { }

  ~LpcAnalyzer(void) { }


  // Apply Hanning weighting to the input data sequence in data_in,
  // put the results in data_out.  Array lengths are assumed as
  // data_in has windowsize+1 points; data_out has windowsize
  // elements.
  void HannWindow(const float* data_in, float* data_out, int windowsize,
                  float preemp);


  // Compute the order+1 autocorrelation lags of the windowsize
  // samples in data_in.  Return the normalized autocorrelation
  // coefficients in autoc.  The rms is returned in rms.
  void Autoc(int windowsize, float* data_in, int order, float* autoc,
             float* rms);


  // Using Durbin's recursion, convert the autocorrelation sequence in autocor
  // to reflection coefficients in refcof and predictor coefficients in lpc.
  // The prediction error energy (gain) is left in *gain.
  // Note: Durbin returns the coefficients in normal sign format.
  // (i.e. lpca[0] is assumed to be = +1.)
  void Durbin(float* autocor, float* refcof, float* lpc, int order,
              float* gain);


  //  Compute the autocorrelations of the order LP coefficients in lpc.
  //  (lpc[0] is assumed to be = 1 and not explicitely accessed.)
  //  The magnitude of lpc is returned in mag.
  //  2* the other autocorrelation coefficients are returned in lpc_auto.
  void PcToAutocorPc(float* lpc, float* lpc_auto, float* mag, int order);


  // Compute the Itakura LPC distance between the model represented
  // by the signal autocorrelation (autoc) and its residual (gain) and
  // the model represented by an LPC autocorrelation (mag, lpc_auto).
  // Both models are of order.
  // r is assumed normalized and r[0]=1 is not explicitely accessed.
  // Values returned by the function are >= 1.
  float ItakuraDistance(int order, float*  lpc_auto, float* mag, float* autoc,
                        float gain);


  // Compute the time-weighted RMS of a size segment of data.  The data
  // is weighted by a window of type w_type before RMS computation.  w_type
  // is decoded above in window().
  float WindowedRms(float* data, int size);


  // Generic autocorrelation LPC analysis of the floating-point
  // sequence in data.
  //
  // int lpc_ord,                /* Analysis order
  //  wsize;                    /* window size in points
  // float noise_floor,  /* To simulate a white noise floor (dB SNR).
  //  *lpca,            /* if non-NULL, return vector for predictors
  //  *ar,              /* if non-NULL, return vector for normalized autoc.
  //  *lpck,            /* if non-NULL, return vector for PARCOR's
  //  *normerr,         /* return scalar for normalized error
  //  *rms,             /* return scalar for energy in preemphasized window
  //  preemp;
  // float *data;           /* input data sequence; assumed to be wsize+1 long
  int ComputeLpc(int lpc_ord, float noise_floor, int wsize, const float* data,
               float* lpca, float* ar, float* lpck, float* normerr, float* rms,
                 float preemp);

  // Use the standard speech analysis formula to determine the
  // appropriate LPC order, given a sample rate.
  static int GetLpcOrder(float sample_rate) {
    return static_cast<int>(2.5 + (sample_rate / 1000.0));
  }

 private:
  // Puts  a time-weighting window of length n in energywind.
  void GetWindow(int n);

  std::vector<float> energywind_;
  std::vector<float> window_;
};

#endif  // _LPC_ANALYZER_H_
