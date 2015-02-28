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
//  Implementation of the LpcAnalyzer class.
//
// Note that this is derived from legacy code written by David Talkin
// before the flood.  Hence the archaic style, etc.

#include "epoch_tracker/lpc_analyzer.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI (3.14159265359)
#endif

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* Generate a Hanning window, if one does not already exist. */
void LpcAnalyzer::HannWindow(const float* din, float* dout, int n,
                             float preemp) {
  int i;
  const float *p;

  // Need to create a new Hanning window? */
  if (window_.size() != static_cast<size_t>(n)) {
    double arg, half = 0.5;
    window_.resize(n);
    for (i = 0, arg = M_PI * 2.0 / n; i < n; ++i)
      window_[i] = (half - half * cos((half + i) * arg));
  }
  /* If preemphasis is to be performed,  this assumes that there are n+1 valid
     samples in the input buffer (din). */
  if (preemp != 0.0) {
    for (i = 0, p = din + 1; i < n; ++i)
      *dout++ = window_[i] * (*p++ - (preemp * *din++));
  } else {
    for (i = 0; i < n; ++i)
      *dout++ = window_[i] * *din++;
  }
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* Place a time-weighting window of  length n in energywind_.
*/
void LpcAnalyzer::GetWindow(int n) {
  //  Need to create a new Hanning window?
  if (energywind_.size() != static_cast<size_t>(n)) {
    double arg = M_PI * 2.0 / n, half = 0.5;
    energywind_.resize(n);
    for (int i = 0; i < n; ++i)
      energywind_[i] = (half - half * cos((half + i) * arg));
  }
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* Compute the pp+1 autocorrelation lags of the windowsize samples in s.
* Return the normalized autocorrelation coefficients in r.
* The rms is returned in e.
*/
void LpcAnalyzer::Autoc(int windowsize, float* s, int p, float* r, float* e) {
  int i, j;
  float *q, *t, sum, sum0;

  for (i = windowsize, q = s, sum0 = 0.0; i--;) {
    sum = *q++;
    sum0 += sum*sum;
  }
  *r = 1.;                      /* r[0] will always = 1.0 */
  if (sum0 == 0.0) {            /* No energy: fake low-energy white noise. */
    *e = 1.;                    /* Arbitrarily assign 1 to rms. */
    /* Now fake autocorrelation of white noise. */
    for (i = 1; i <= p; i++) {
      r[i] = 0.;
    }
    return;
  }
  *e = sqrt(sum0 / windowsize);
  sum0 = 1.0 / sum0;
  for (i = 1; i <= p; i++) {
    for (sum = 0.0, j = windowsize - i, q = s, t = s + i; j--;)
      sum += (*q++) * (*t++);
    *(++r) = sum * sum0;  // normalizing by the inverse energy
  }
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* Using Durbin's recursion, convert the autocorrelation sequence in r
* to reflection coefficients in k and predictor coefficients in a.
* The prediction error energy (gain) is left in *ex.
* Note: Durbin returns the coefficients in normal sign format.
*       (i.e. a[0] is assumed to be = +1.)
*/
void LpcAnalyzer::Durbin(float* r, float* k, float* a, int p, float* ex) {
  float  bb[BIGSORD];
  int i, j;
  float e, s, *b = bb;

  e = *r;
  *k = -r[1] / e;
  *a = *k;
  e *= (1.0 - (*k) * (*k));
  for (i = 1; i < p; i++) {
    s = 0;
    for (j = 0; j < i; j++) {
      s -= a[j] * r[i - j];
    }
    k[i] = (s - r[i + 1]) / e;
    a[i] = k[i];
    for (j = 0; j <= i; j++) {
      b[j] = a[j];
    }
    for (j = 0; j < i; j++) {
      a[j] += k[i] * b[i - j - 1];
    }
    e *= (1.0 - (k[i] * k[i]));
  }
  *ex = e;
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*  Compute the autocorrelations of the p LP coefficients in a. 
*  (a[0] is assumed to be = 1 and not explicitely accessed.)
*  The magnitude of a is returned in c.
*  2* the other autocorrelation coefficients are returned in b.
*/
void LpcAnalyzer::PcToAutocorPc(float* a, float* b, float* c, int p) {
  float  s, *ap, *a0;
  int  i, j;

  for (s = 1., ap = a, i = p; i--; ap++)
    s += *ap * *ap;

  *c = s;
  for (i = 1; i <= p; i++) {
    s = a[i - 1];
    for (a0 = a, ap = a + i, j = p - i; j--;)
      s += (*a0++ * *ap++);
    *b++ = 2.0 * s;
  }
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* Compute the Itakura LPC distance between the model represented
* by the signal autocorrelation (r) and its residual (gain) and
* the model represented by an LPC autocorrelation (c, b).
* Both models are of order p.
* r is assumed normalized and r[0]=1 is not explicitely accessed.
* Values returned by the function are >= 1.
*/
float LpcAnalyzer::ItakuraDistance(int p, float*  b, float* c, float* r,
                                    float gain) {
  float s;

  for (s = *c; p--;)
    s += *r++ * *b++;

  return s / gain;
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* Compute the time-weighted RMS of a size segment of data.  The data
* is weighted by a Hanning window before RMS computation.
*/
float LpcAnalyzer::WindowedRms(float* data, int size) {
  float sum, f;
  int i;

  GetWindow(size);
  for (i = 0, sum = 0.0; i < size; i++) {
    f = energywind_[i] * (*data++);
    sum += f * f;
  }
  return sqrt(sum / size);
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// Generic autocorrelation LPC analysis of the floating-point
// sequence in data.
//
// int lpc_ord,          /* Analysis order */
//  wsize,                      /* window size in points */
//  type;               /* window type (decoded in window() above) */
// float noise_floor,    /* Simulated white noise floor in dB (SNR). */
//  *lpca,              /* if non-NULL, return vector for predictors */
//  *ar,                /* if non-NULL, return vector for normalized autoc. */
//  *lpck,              /* if non-NULL, return vector for PARCOR's */
//  *normerr,           /* return scalar for normalized error */
//  *rms,               /* return scalar for energy in preemphasized window */
//  preemp;
// float *data;  /* input data sequence; assumed to be wsize+1 long */
int LpcAnalyzer::ComputeLpc(int lpc_ord, float noise_floor, int wsize,
                            const float* data, float* lpca, float* ar,
                            float* lpck, float* normerr, float* rms,
                            float preemp) {
  float rho[BIGSORD+1], k[BIGSORD], a[BIGSORD+1], *r, *kp, *ap, en, er;

  if ((wsize <= 0) || (!data) || (lpc_ord > BIGSORD))
    return false;

  float *dwind = new float[wsize];

  HannWindow(data, dwind, wsize, preemp);
  if (!(r = ar)) r = rho;       /* Permit optional return of the various */
  if (!(kp = lpck)) kp = k;     /* coefficients and intermediate results. */
  if (!(ap = lpca)) ap = a;
  Autoc(wsize, dwind, lpc_ord, r, &en);
  if (noise_floor > 1.0) {  // Add some to the diagonal to simulate white noise.
    int i;
    float ffact;
    ffact = 1.0 / (1.0 + exp((-noise_floor / 20.0) * log(10.0)));
    for (i = 1; i <= lpc_ord; i++)
      rho[i] = ffact * r[i];
    *rho = *r;
    r = rho;
    if (ar) {
      for (i = 0; i <= lpc_ord; i++)
        ar[i] = r[i];
    }
  }
  Durbin(r, kp, ap + 1, lpc_ord, &er);
  float wfact = .612372;  // ratio of Hanning RMS to rectangular RMS
  ap[0] = 1.0;
  if (rms)
    *rms = en / wfact;
  if (normerr)
    *normerr = er;
  delete [] dwind;
  return true;
}
