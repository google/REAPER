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
// Author: dtalkin@google.com (David Talkin)

// Implementation of the EpochTracker class.  This does all of the
// processing necessary to estimate the F0, voicing state and epochs
// (glottal-closure instants) in human speech signals.  See
// epoch_tracker.h for details.

#include "epoch_tracker/epoch_tracker.h"

#include <string>
#include <vector>

#include "epoch_tracker/fd_filter.h"
#include "epoch_tracker/lpc_analyzer.h"
#include "epoch_tracker/fft.h"

const int kMinSampleRate = 6000;

EpochTracker::EpochTracker(void) : sample_rate_(-1.0) {
  SetParameters();
}

EpochTracker::~EpochTracker(void) {
  CleanUp();
}

static inline int32_t RoundUp(float val) {
  return static_cast<int32_t>(val + 0.5);
}

void EpochTracker::CleanUp(void) {
  for (size_t i = 0; i < resid_peaks_.size(); ++i) {
    for (size_t j = 0; j < resid_peaks_[i].future.size(); ++j) {
      delete resid_peaks_[i].future[j];
    }
  }
  resid_peaks_.clear();
  output_.clear();
  best_corr_.clear();
}

void EpochTracker::SetParameters(void) {
  // Externally-settable control parameters:
  // Period for the returned F0 signal.
  external_frame_interval_ = kExternalFrameInterval;
  do_highpass_ = kDoHighpass;  // Enables highpassing of input signal.
  // Enables Hilbert transformation of the input data.
  do_hilbert_transform_ = kDoHilbertTransform;
  max_f0_search_ = kMaxF0Search;  // Maximum F0 to search for.
  min_f0_search_ = kMinF0Search;  // Minimum F0 to search for.
  // Pulse spacing to use in unvoiced regions of the returned epoch signal.
  unvoiced_pulse_interval_ = kUnvoicedPulseInterval;
  debug_name_ = kDebugName;  // base path for all debugging signals.

  // Internal feature-computation parameters:
  // For internal feature computations
  internal_frame_interval_ = kInternalFrameInterval;
  // for the high-pass filter
  corner_frequency_ = 80.0;
  filter_duration_ = 0.05;
  // for the LPC inverse filter.
  frame_duration_ = 0.02;  // window size (sec)
  lpc_frame_interval_ = 0.01;  // (sec)
  preemphasis_ = 0.98;  // preemphasis for LPC analysis
  noise_floor_ = 70.0;  // SNR in dB simulated during LPC analysis.
  // for computing LPC residual peak quality.
  peak_delay_ = 0.0004;  // for measuring prominence
  skew_delay_ = 0.00015;  // for measuring shape
  peak_val_wt_ = 0.1;
  peak_prominence_wt_ = 0.3;
  peak_skew_wt_ = 0.1;
  peak_quality_floor_ = 0.01;
  // for computing voice-transition pseudo-probabilities
  time_span_ = 0.020;  // the interval (sec) centered on the
  // measurement point, used to compute parameter
  // deltas.
  level_change_den_ = 30.0;  // max. dB level change expected over
  // time_span_ for bandpassed RMS for
  // computing pseudo-probability of
  // voicing.
  min_rms_db_ = 20.0;  // level floor in dB
  // window size for computing amplitude-normalizing RMS
  ref_dur_ = 0.02;
  // low and high frequency limits for bandpassed RMS used in voicing indicator
  min_freq_for_rms_ = 100.0;
  max_freq_for_rms_ = 1000.0;
  // duration of integrator for bandpassed RMS
  rms_window_dur_ = 0.025;
  // window duration, in seconds, for NCCF computations
  correlation_dur_ = 0.0075;
  // ignore any NCCF peaks less than this
  correlation_thresh_ = 0.2;

  // Parametrs used by the dynamic-programming tracker:
  // reward for inserting another period
  reward_ = -1.5;
  // weight given to deviation of inter-pulse interval from the
  // closest NCCF peak lag
  period_deviation_wt_ = 1.0;
  // weight given to the quality of the residual peak
  peak_quality_wt_ = 1.3;
  // cost of the unvoiced hypothesis
  unvoiced_cost_ = kUnvoicedCost;
  // cost of high NCCF values in hypothetical unvoiced regions
  nccf_uv_peak_wt_ = 0.9;
  // weight given to period length
  period_wt_ = 0.0002;
  // weight given to the pseudo-probability of voicing feature
  level_wt_ = 0.8;
  // weight given to period-length differences between adjacent periods.
  freq_trans_wt_ = 1.8;
  // cost of switching between voicing states; modulated by voicing
  // onset/offset probs.
  voice_transition_factor_ = 1.4;

  // Parameters used to generate final outputs:
  // pad time in seconds to add to the last measured period during
  // output of periodically-resampled data
  endpoint_padding_ = 0.01;
}

bool EpochTracker::Init(const int16_t* input, int32_t n_input, float sample_rate,
                        float min_f0_search, float max_f0_search,
                        bool do_highpass, bool do_hilbert_transform) {
  if (input && (sample_rate > 6000.0) && (n_input > (sample_rate * 0.05)) &&
      (min_f0_search < max_f0_search) && (min_f0_search > 0.0)) {
    CleanUp();
    min_f0_search_ = min_f0_search;
    max_f0_search_ = max_f0_search;
    sample_rate_ = sample_rate;
    int16_t* input_p = const_cast<int16_t *>(input);
    if (do_highpass) {
      input_p = HighpassFilter(input_p, n_input, sample_rate,
                               corner_frequency_, filter_duration_);
    }
    signal_.resize(n_input);
    if (do_hilbert_transform) {
      HilbertTransform(input_p, n_input, &(signal_.front()));
    } else {
      for (int32_t i = 0; i < n_input; ++i) {
        signal_[i] = input_p[i];
      }
    }
    if (input_p != input) {
      delete [] input_p;
    }
    return true;
  }
  return false;
}

void EpochTracker::HilbertTransform(int16_t* input, int32_t n_input,
                                    float* output) {
  FFT ft = FFT(FFT::fft_pow2_from_window_size(n_input));
  int32_t n_fft = ft.get_fftSize();
  float* re = new float[n_fft];
  float* im = new float[n_fft];
  for (int i = 0; i < n_input; ++i) {
    re[i] = input[i];
    im[i] = 0.0;
  }
  for (int i = n_input; i < n_fft; ++i) {
    re[i] = 0.0;
    im[i] = 0.0;
  }
  ft.fft(re, im);
  for (int i = 1; i < n_fft/2; ++i) {
    float tmp = im[i];
    im[i] = -re[i];
    re[i] = tmp;
  }
  re[0] = im[0] = 0.0;
  for (int i = n_fft/2 + 1; i < n_fft; ++i) {
    float tmp = im[i];
    im[i] = re[i];
    re[i] = -tmp;
  }
  ft.ifft(re, im);
  for (int i = 0; i < n_input; ++i) {
    output[i] = re[i] / n_fft;
  }
  delete [] re;
  delete [] im;
}


int16_t* EpochTracker::HighpassFilter(int16_t* input, int32_t n_input,
                                      float sample_rate, float corner_freq,
				      float fir_duration) {
  FdFilter filter(sample_rate, corner_freq, true, fir_duration, false);
  int16_t* filtered_data = new int16_t[n_input];
  int32_t max_buffer_size = filter.GetMaxInputSize();
  int32_t to_process = n_input;
  bool start = true;
  bool end = false;
  int32_t input_index = 0;
  int32_t output_index = 0;
  while (to_process > 0) {
    int32_t to_send = to_process;
    if (to_send > max_buffer_size) {
      to_send = max_buffer_size;
    } else {
      end = true;
    }
    int32_t samples_returned = filter.FilterArray(input + input_index, to_send,
                                                  start, end,
                                                  filtered_data + output_index,
                                                  n_input - output_index);
    input_index += to_send;
    to_process -= to_send;
    output_index += samples_returned;
    start = false;
  }
  return filtered_data;
}


static float LpcDcGain(float* lpc, int32_t order) {
  float sum = 0.0;
  for (int32_t i = 0; i <= order; ++i) {
    sum += lpc[i];
  }
  if (sum > 0.0) {
    return sum;
  } else {
    return 1.0;
  }
}


static void MakeDeltas(float* now, float* next, int32_t size, int32_t n_steps,
                       float* deltas) {
  for (int32_t i = 0; i < size; ++i) {
    deltas[i] = (next[i] - now[i]) / n_steps;
  }
}


bool EpochTracker::GetLpcResidual(const std::vector<float>& input, float sample_rate,
                                  std::vector<float>* output) {
  int32_t n_input = input.size();
  if (!((n_input > 0) && (sample_rate > 0.0) && output)) {
    return false;
  }
  output->resize(n_input);
  int32_t frame_step = RoundUp(sample_rate * lpc_frame_interval_);
  int32_t frame_size = RoundUp(sample_rate * frame_duration_);
  int32_t n_frames = 1 + ((n_input - frame_size) / frame_step);
  int32_t n_analyzed = ((n_frames - 1) * frame_step) + frame_size;
  // Must have one more than frame size to do a complete frame.
  if (n_analyzed <= n_input) {
    n_frames--;
    if (n_frames <= 0) {
      return false;
    }
  }
  LpcAnalyzer lp;
  int32_t order = lp.GetLpcOrder(sample_rate);
  float* lpc = new float[order + 1];
  float* old_lpc = new float[order + 1];
  float* delta_lpc = new float[order + 1];
  float norm_error = 0.0;
  float preemp_rms = 0.0;

#define  RELEASE_MEMORY() {                     \
    delete [] lpc;                              \
    delete [] old_lpc;                          \
    delete [] delta_lpc;                        \
  }

  if (!lp.ComputeLpc(order, noise_floor_, frame_size, &(input.front()),
                     old_lpc, NULL, NULL, &norm_error, &preemp_rms,
                     preemphasis_)) {
    RELEASE_MEMORY();
    return false;
  }
  for (int32_t i = 0; i <= order; ++i) {
    delta_lpc[i] = 0.0;
    (*output)[i] = 0.0;
  }
  float old_gain = LpcDcGain(old_lpc, order);
  float new_gain = 1.0;
  int32_t n_to_filter = (frame_size / 2) - order;  // How many samples
  // to process before
  // computing the next
  // LPC frame.
  int32_t input_p = 0;  // Where to get the next frame for LPC analysis
  int32_t output_p = order;  // where to store output samples.
  int32_t proc_p = 0;  // Where to pick up samples for input to the filter

  // Main processing loop:
  // Compute a new frame of LPC
  // Compute the DC gain for the new LPC
  // Compute delta DC gain.
  // Compute the LPC deltas.
  // For each point in the frame:
  //   Use old_lpc to produce an output point.
  //   Update the old LPCs and the DC gain
  // As soon as the center of the current frame is reached, compute
  // the LPC for the next frame.
  for ( ; n_frames > 0; --n_frames, input_p += frame_step,
            n_to_filter = frame_step) {
    if (!lp.ComputeLpc(order, noise_floor_, frame_size,
                       (&(input.front())) + input_p, lpc, NULL, NULL,
                       &norm_error, &preemp_rms, preemphasis_)) {
      RELEASE_MEMORY();
      return false;
    }
    new_gain = LpcDcGain(lpc, order);
    float delta_gain = (new_gain - old_gain) / n_to_filter;
    MakeDeltas(old_lpc, lpc, order+1, n_to_filter, delta_lpc);
    for (int32_t sample = 0; sample < n_to_filter; ++sample, ++proc_p,
             ++output_p) {
      float sum = 0.0;
      int32_t mem = proc_p;
      for (int32_t k = order; k > 0; --k, ++mem) {
        sum += (old_lpc[k] * input[mem]);
        old_lpc[k] += delta_lpc[k];
      }
      sum += input[mem];  // lpc[0] is always 1.0
      (*output)[output_p] = sum / old_gain;
      old_gain += delta_gain;
    }
  }
  RELEASE_MEMORY();
  return true;
}

// Note that GetResidualPulses assumes the LPC residual is in the
// "correct" polarity, with the GCI pulses of interest being negative
// pulses with a gradual fall and an abrupt rise.
void EpochTracker::GetResidualPulses(void) {
  int32_t peak_ind = RoundUp(peak_delay_ * sample_rate_);
  int32_t skew_ind = RoundUp(skew_delay_ * sample_rate_);
  float min_peak = -1.0;  // minimum value that will be considered as a peak
  int32_t limit = norm_residual_.size() - peak_ind;
  resid_peaks_.resize(0);
  peaks_debug_.resize(residual_.size());
  for (size_t i = 0; i < peaks_debug_.size(); ++i) {
    peaks_debug_[i] = 0.0;
  }
  for (int32_t i = peak_ind; i < limit; ++i) {
    float val = norm_residual_[i];
    if (val > min_peak) {
      continue;
    }
    if ((norm_residual_[i-1] > val) && (val <= norm_residual_[i+1])) {
      float vm_peak = norm_residual_[i - peak_ind];
      float vp_peak = norm_residual_[i + peak_ind];
      if ((vm_peak < val) || (vp_peak < val)) {
        continue;
      }
      float vm_skew = norm_residual_[i - skew_ind];
      float vp_skew = norm_residual_[i + skew_ind];
      float sharp = (0.5 * (vp_peak + vm_peak)) - val;
      float skew = -(vm_skew - vp_skew);
      ResidPeak p;
      p.resid_index = i;
      float time = static_cast<float>(i) / sample_rate_;
      p.frame_index = RoundUp(time / internal_frame_interval_);
      if (p.frame_index >= n_feature_frames_) {
        p.frame_index = n_feature_frames_ - 1;
      }
      p.peak_quality = (-val * peak_val_wt_) + (skew * peak_skew_wt_) +
          (sharp * peak_prominence_wt_);
      if (p.peak_quality < peak_quality_floor_) {
        p.peak_quality = peak_quality_floor_;
      }
      resid_peaks_.push_back(p);
      peaks_debug_[i] = p.peak_quality;
    }
  }
}


void EpochTracker::GetVoiceTransitionFeatures(void) {
  int32_t frame_offset = RoundUp(0.5 * time_span_ / internal_frame_interval_);
  if (frame_offset <= 0) {
    frame_offset = 1;
  }
  voice_onset_prob_.resize(n_feature_frames_);
  voice_offset_prob_.resize(n_feature_frames_);
  int32_t limit = n_feature_frames_ - frame_offset;
  for (int32_t frame = frame_offset; frame < limit; ++frame) {
    float delta_rms = (bandpassed_rms_[frame + frame_offset] -
                       bandpassed_rms_[frame - frame_offset]) / level_change_den_;
    if (delta_rms > 1.0) {
      delta_rms = 1.0;
    } else {
      if (delta_rms < -1.0) {
        delta_rms = -1.0;
      }
    }
    float prob_onset = delta_rms;
    float prob_offset = -prob_onset;
    if (prob_onset > 1.0) {
      prob_onset = 1.0;
    } else {
      if (prob_onset < 0.0) {
        prob_onset = 0.0;
      }
    }
    if (prob_offset > 1.0) {
      prob_offset = 1.0;
    } else {
      if (prob_offset < 0.0) {
        prob_offset = 0.0;
      }
    }
    voice_onset_prob_[frame] = prob_onset;
    voice_offset_prob_[frame] = prob_offset;
  }
  // Just set the onset and offset probs to zero in the end zones.
  for (int32_t frame = 0; frame < frame_offset; ++frame) {
    int32_t bframe = n_feature_frames_ - 1 - frame;
    voice_onset_prob_[frame] = voice_offset_prob_[frame] = 0.0;
    voice_onset_prob_[bframe] = voice_offset_prob_[bframe] = 0.0;
  }
}


void EpochTracker::GetRmsVoicingModulator(void) {
  float min_val = bandpassed_rms_[0];
  float max_val = min_val;

  prob_voiced_.resize(bandpassed_rms_.size());
  // Find the max and min over the whole RMS array.  Scale and offset
  // the RMS values to all fall in th range of 0.0 to 1.0.
  for (size_t i = 1; i < bandpassed_rms_.size(); ++i) {
    float val = bandpassed_rms_[i];
    if (val < min_val) {
      min_val = val;
    } else {
      if (val > max_val) {
        max_val = val;
      }
    }
  }
  if (min_val < min_rms_db_) {
    min_val = min_rms_db_;
  }
  float range = max_val - min_val;
  if (range < 1.0) {
    range = 1.0;
  }
  for (size_t i = 0; i < bandpassed_rms_.size(); ++i) {
    prob_voiced_[i] = (bandpassed_rms_[i] - min_val) / range;
    if (prob_voiced_[i] < 0.0) {
      prob_voiced_[i] = 0.0;
    }
  }
}


int32_t EpochTracker::FindNccfPeaks(const std::vector<float>& input, float thresh,
                                    std::vector<int16_t>* output) {
  int32_t limit = input.size() - 1;
  uint32_t n_peaks = 0;
  float max_val = 0.0;
  int16_t max_index = 1;
  int16_t max_out_index = 0;
  output->resize(0);
  for (int16_t i = 1; i < limit; ++i) {
    float val = input[i];
    if ((val > thresh) && (val > input[i-1]) && (val >= input[i+1])) {
      if (val > max_val) {
        max_val = val;
        max_out_index = n_peaks;
        max_index = i;
      }
      n_peaks++;
      output->push_back(i);
    }
  }
  //  Be sure the highest peak is the first one in the array.
  if ((n_peaks > 1) && (max_out_index > 0)) {
    int16_t hold = (*output)[0];
    (*output)[0] = (*output)[max_out_index];
    (*output)[max_out_index] = hold;
  } else {
    if (n_peaks <= 0) {
      n_peaks = 1;
      output->push_back(max_index);
    }
  }
  return n_peaks;
}


void EpochTracker::CrossCorrelation(const std::vector<float>& data, int32_t start,
                                    int32_t first_lag, int32_t n_lags,
                                    int32_t size, std::vector<float>* corr) {
  const float* input = (&(data.front())) + start;
  corr->resize(n_lags);
  float energy = 0.0;  // Zero-lag energy part of the normalizer.
  for (int32_t i = 0; i < size; ++i) {
    energy += input[i] * input[i];
  }
  if (energy == 0.0) {  // Bail out if no energy is found.
    for (int32_t i = 0; i < n_lags; ++i) {
      (*corr)[i] = 0.0;
    }
    return;
  }
  int32_t limit = first_lag + size;
  double lag_energy = 0.0;  // Energy at the period hypothesis lag.
  for (int32_t i = first_lag; i < limit; ++i) {
    lag_energy += input[i] * input[i];
  }
  int32_t last_lag = first_lag + n_lags;
  int32_t oind = 0;  // Index for storing output values.
  for (int32_t lag = first_lag; lag < last_lag; ++lag, ++oind) {
    float sum = 0.0;
    int32_t lag_ind = lag;
    for (int32_t i = 0; i < size; ++i, ++lag_ind) {
      sum += input[i] * input[lag_ind];
    }
    if (lag_energy <= 0.0)
      lag_energy = 1.0;
    (*corr)[oind] = sum / sqrt(energy * lag_energy);
    lag_energy -= input[lag] * input[lag];  // Discard old sample.
    lag_energy += input[lag_ind] * input[lag_ind];  // Pick up the new sample.
  }
  return;
}


void EpochTracker::GetPulseCorrelations(float window_dur, float peak_thresh) {
  first_nccf_lag_ = RoundUp(sample_rate_ / max_f0_search_);
  int32_t max_lag = RoundUp(sample_rate_ / min_f0_search_);
  n_nccf_lags_ = max_lag - first_nccf_lag_;
  int32_t window_size = RoundUp(window_dur * sample_rate_);
  int32_t half_wind = window_size / 2;
  int32_t frame_size = window_size + max_lag;

  std::vector<float> mixture;
  mixture.resize(residual_.size());
  const float kMinCorrelationStep = 0.001;  // Pulse separation
  // before computing new
  // correlation values.
  const float kResidFract = 0.7;  // Fraction of the residual to use.
  const float kPcmFract = 1.0 - kResidFract;  // Fraction of the input to use.
  for (size_t i = 0; i < residual_.size(); ++i) {
    mixture[i] = (kResidFract * residual_[i]) + (kPcmFract * signal_[i]);
  }

  int32_t min_step = RoundUp(sample_rate_ * kMinCorrelationStep);
  int32_t old_start = - (2.0 * min_step);
  for (size_t peak = 0; peak < resid_peaks_.size(); ++peak) {
    int32_t start = resid_peaks_[peak].resid_index - half_wind;
    if (start < 0) {
      start = 0;
    }
    size_t end = start + frame_size;
    if ((end >= mixture.size()) || ((start - old_start) < min_step)) {
      resid_peaks_[peak].nccf = resid_peaks_[peak - 1].nccf;
      resid_peaks_[peak].nccf_periods = resid_peaks_[peak - 1].nccf_periods;
    } else {
      CrossCorrelation(mixture, start, first_nccf_lag_, n_nccf_lags_,
                       window_size, &(resid_peaks_[peak].nccf));
      FindNccfPeaks(resid_peaks_[peak].nccf, peak_thresh,
                    &(resid_peaks_[peak].nccf_periods));
      // Turn the peak indices from FindNccfPeaks into NCCF period hyps.
      for (size_t i = 0; i < resid_peaks_[peak].nccf_periods.size(); ++i) {
        resid_peaks_[peak].nccf_periods[i] += first_nccf_lag_;
      }
      old_start = start;
    }
  }
}


void EpochTracker::Window(const std::vector<float>& input, int32_t offset, size_t size,
                          float* output) {
  if (size != window_.size()) {
    window_.resize(size);
    float arg = 2.0 * M_PI / size;
    for (size_t i = 0; i < size; ++i) {
      window_[i] = 0.5 - (0.5 * cos((i + 0.5) * arg));
    }
  }
  const float* data = (&(input.front())) + offset;
  for (size_t i = 0; i < size; ++i) {
    output[i] = data[i] * window_[i];
  }
}


bool EpochTracker::GetBandpassedRmsSignal(const std::vector<float>& input,
                                          float sample_rate,
                                          float low_limit, float high_limit,
                                          float frame_interval,
                                          float frame_dur,
                                          std::vector<float>* output_rms) {
  size_t frame_step = RoundUp(sample_rate * frame_interval);
  size_t frame_size = RoundUp(sample_rate * frame_dur);
  size_t n_frames = 1 + ((input.size() - frame_size) / frame_step);
  if (n_frames < 2) {
    fprintf(stderr, "input too small (%d) in GetBandpassedRmsSignal\n",
            static_cast<int>(input.size()));
    output_rms->resize(0);
    return false;
  }
  output_rms->resize(n_frames);
  FFT ft(FFT::fft_pow2_from_window_size(frame_size));
  int32_t fft_size = ft.get_fftSize();
  int32_t first_bin = RoundUp(fft_size * low_limit / sample_rate);
  int32_t last_bin = RoundUp(fft_size * high_limit / sample_rate);
  float* re = new float[fft_size];
  float* im = new float[fft_size];
  size_t first_frame = frame_size / (2 * frame_step);
  if ((first_frame * 2 * frame_step) < frame_size) {
    first_frame++;
  }
  for (size_t frame = first_frame; frame < n_frames; ++frame) {
    Window(input, (frame - first_frame) * frame_step, frame_size, re);
    for (size_t i = 0; i < frame_size; ++i) {
      im[i] = 0.0;
    }
    for (int32_t i = frame_size; i < fft_size; ++i) {
      re[i] = im[i] = 0.0;
    }
    ft.fft(re, im);
    float rms = 20.0 *
        log10(1.0 + ft.get_band_rms(re, im, first_bin, last_bin));
    (*output_rms)[frame] = rms;
    if (frame == first_frame) {
      for (size_t bframe = 0; bframe < first_frame; ++bframe) {
        (*output_rms)[bframe] = rms;
      }
    }
  }
  delete [] re;
  delete [] im;
  return true;
}


void EpochTracker::GetSymmetryStats(const std::vector<float>& data, float* pos_rms,
                                    float* neg_rms, float* mean) {
  int32_t n_input = data.size();
  double p_sum = 0.0;
  double n_sum = 0.0;
  double sum = 0.0;
  int32_t n_p = 0;
  int32_t n_n = 0;
  for (int32_t i = 0; i < n_input; ++i) {
    sum += data[i];
  }
  *mean = sum / n_input;
  for (int32_t i = 0; i < n_input; ++i) {
    double val = data[i] - *mean;
    if (val > 0.0) {
      p_sum += (val * val);
      n_p++;
    } else {
      if (val < 0.0) {
        n_sum += (val * val);
        n_n++;
      }
    }
  }
  *pos_rms = sqrt(p_sum / n_p);
  *neg_rms = sqrt(n_sum / n_n);
}


void EpochTracker::NormalizeAmplitude(const std::vector<float>& input,
                                      float sample_rate,
                                      std::vector<float>* output) {
  int32_t n_input = input.size();
  int32_t ref_size = RoundUp(sample_rate * ref_dur_);
  std::vector<float> wind;

  output->resize(n_input);
  // Just calling Window here to create a Hann window in window_.
  Window(input, 0, ref_size, &(output->front()));
  int32_t ref_by_2 = ref_size / 2;
  int32_t frame_step = RoundUp(sample_rate * internal_frame_interval_);
  int32_t limit = n_input - ref_size;
  int32_t frame_limit = ref_by_2;
  int32_t data_p = 0;
  int32_t frame_p = 0;
  double old_inv_rms = 0.0;
  while (frame_p < limit) {
    double ref_energy = 1.0;  // to prevent divz
    for (int32_t i = 0; i < ref_size; ++i) {
      double val = window_[i] * input[i + frame_p];
      ref_energy += (val * val);
    }
    double inv_rms = sqrt(static_cast<double>(ref_size) / ref_energy);
    double delta_inv_rms = 0.0;
    if (frame_p > 0) {
      delta_inv_rms = (inv_rms - old_inv_rms) / frame_step;
    } else {
      old_inv_rms = inv_rms;
    }
    for (int i = 0; i < frame_limit; ++i, ++data_p) {
      (*output)[data_p] = input[data_p] * old_inv_rms;
      old_inv_rms += delta_inv_rms;
    }
    frame_limit = frame_step;
    frame_p += frame_step;
  }
  for ( ; data_p < n_input; ++data_p) {
    (*output)[data_p] = input[data_p] * old_inv_rms;
  }
}

bool EpochTracker::ComputePolarity(int *polarity) {
  if (sample_rate_ <= 0.0) {
    fprintf(stderr, "EpochTracker not initialized in ComputeFeatures\n");
    return false;
  }
  if (!GetBandpassedRmsSignal(signal_, sample_rate_, min_freq_for_rms_,
                              max_freq_for_rms_, internal_frame_interval_,
                              rms_window_dur_, &bandpassed_rms_)) {
    fprintf(stderr, "Failure in GetBandpassedRmsSignal\n");
    return false;
  }
  if (!GetLpcResidual(signal_, sample_rate_, &residual_)) {
    fprintf(stderr, "Failure in GetLpcResidual\n");
    return false;
  }
  float mean = 0.0;
  GetSymmetryStats(residual_, &positive_rms_, &negative_rms_, &mean);
  *polarity = -1;
  if (positive_rms_ > negative_rms_) {
    *polarity = 1;
  }
  return true;
}

bool EpochTracker::ComputeFeatures(void) {
  if (sample_rate_ <= 0.0) {
    fprintf(stderr, "EpochTracker not initialized in ComputeFeatures\n");
    return false;
  }
  if (!GetBandpassedRmsSignal(signal_, sample_rate_, min_freq_for_rms_,
                              max_freq_for_rms_, internal_frame_interval_,
                              rms_window_dur_, &bandpassed_rms_)) {
    fprintf(stderr, "Failure in GetBandpassedRmsSignal\n");
    return false;
  }
  if (!GetLpcResidual(signal_, sample_rate_, &residual_)) {
    fprintf(stderr, "Failure in GetLpcResidual\n");
    return false;
  }
  n_feature_frames_ = bandpassed_rms_.size();
  float mean = 0.0;
  GetSymmetryStats(residual_, &positive_rms_, &negative_rms_, &mean);
  fprintf(stdout, "Residual symmetry: P:%f  N:%f  MEAN:%f\n",
	  positive_rms_, negative_rms_, mean);
  if (positive_rms_ > negative_rms_) {
    fprintf(stdout, "Inverting signal\n");
    for (size_t i = 0; i < residual_.size(); ++i) {
      residual_[i] = -residual_[i];
      signal_[i] = -signal_[i];
    }
  }
  NormalizeAmplitude(residual_, sample_rate_, &norm_residual_);
  GetResidualPulses();
  GetPulseCorrelations(correlation_dur_, correlation_thresh_);
  GetVoiceTransitionFeatures();
  GetRmsVoicingModulator();
  return true;
}


bool EpochTracker::TrackEpochs(void) {
  CreatePeriodLattice();
  DoDynamicProgramming();
  return BacktrackAndSaveOutput();
}


void EpochTracker::CreatePeriodLattice(void) {
  int32_t low_period = RoundUp(sample_rate_ / max_f0_search_);
  int32_t high_period = RoundUp(sample_rate_ / min_f0_search_);
  int32_t total_cands = 0;

  //  For each pulse in the normalized residual...
  for (size_t peak = 0; peak < resid_peaks_.size(); ++peak) {
    size_t frame_index = resid_peaks_[peak].frame_index;
    size_t resid_index = resid_peaks_[peak].resid_index;
    int32_t min_period = resid_index + low_period;
    int32_t max_period = resid_index + high_period;
    float lowest_cost = 1.0e30;
    float time = resid_index / sample_rate_;
    int32_t best_nccf_period = resid_peaks_[peak].nccf_periods[0];
    float best_cc_val =
        resid_peaks_[peak].nccf[best_nccf_period - first_nccf_lag_];
    best_corr_.push_back(time);
    best_corr_.push_back(best_cc_val);
    EpochCand* uv_cand = new EpochCand;  // pre-allocate an unvoiced candidate.
    uv_cand->voiced = false;
    uv_cand->start_peak = peak;
    uv_cand->cost_sum = 0.0;
    uv_cand->local_cost = 0.0;
    uv_cand->best_prev_cand = -1;
    int32_t next_cands_created = 0;
    // For each of the next residual pulses in the search range...
    for (size_t npeak = peak + 1; npeak < resid_peaks_.size(); ++npeak) {
      int32_t iperiod = resid_peaks_[npeak].resid_index - resid_index;
      if (resid_peaks_[npeak].resid_index >= min_period) {
        float fperiod = iperiod;
        // Find the NCCF period that most closely matches.
        int32_t cc_peak = 0;
        float min_period_diff = fabs(log(fperiod / best_nccf_period));
        for (size_t cc_peak_ind = 1;
             cc_peak_ind < resid_peaks_[peak].nccf_periods.size();
             ++cc_peak_ind) {
          int32_t nccf_period =  resid_peaks_[peak].nccf_periods[cc_peak_ind];
          float test_diff = fabs(log(fperiod / nccf_period));
          if (test_diff < min_period_diff) {
            min_period_diff = test_diff;
            cc_peak = cc_peak_ind;
          }
        }
        // Generate a forward-period candidate.  Grade the candidate
        // on closeness to a NCCF period hyp, value of the NCCF,
        // values of the candidate endpoint peaks.
        EpochCand* v_cand = new EpochCand;
        v_cand->voiced = true;
        v_cand->period = iperiod;
        int32_t cc_index = iperiod - first_nccf_lag_;
        float cc_value = 0.0;
        // If this period is in the normal search range, retrieve the
        // actual NCCF value for that lag.
        if ((cc_index >= 0) && (cc_index < n_nccf_lags_)) {
          cc_value = resid_peaks_[peak].nccf[cc_index];
        } else {  // punt and use the "closest" nccf peak
          int32_t peak_cc_index = resid_peaks_[peak].nccf_periods[cc_peak] -
              first_nccf_lag_;
          cc_value =  resid_peaks_[peak].nccf[peak_cc_index];
        }
        float per_dev_cost = period_deviation_wt_ * min_period_diff;
        float level_cost = level_wt_ * (1.0 - prob_voiced_[frame_index]);
        float period_cost = fperiod * period_wt_;
        float peak_qual_cost = peak_quality_wt_ /
            (resid_peaks_[npeak].peak_quality + resid_peaks_[peak].peak_quality);
        float local_cost =  (1.0 - cc_value) + per_dev_cost + peak_qual_cost +
            level_cost + period_cost + reward_;
        v_cand->local_cost = local_cost;
        if (local_cost < lowest_cost) {
          lowest_cost = local_cost;
          // Evaluate this best voiced period as an unvoiced
          // hypothesis.  (There are always plenty of poor
          // voiced candidates!)
          uv_cand->period = iperiod;
          level_cost = level_wt_ * prob_voiced_[frame_index];
          uv_cand->local_cost = (nccf_uv_peak_wt_ * cc_value) +
              level_cost + unvoiced_cost_ + reward_;
          uv_cand->end_peak = npeak;
          uv_cand->closest_nccf_period =
              resid_peaks_[peak].nccf_periods[cc_peak];
        }
        v_cand->start_peak = peak;
        v_cand->end_peak = npeak;
        v_cand->closest_nccf_period = resid_peaks_[peak].nccf_periods[cc_peak];
        v_cand->cost_sum = 0.0;
        v_cand->best_prev_cand = -1;
        resid_peaks_[peak].future.push_back(v_cand);
        resid_peaks_[npeak].past.push_back(v_cand);
        total_cands++;
        next_cands_created++;
        if (resid_peaks_[npeak].resid_index >= max_period) {
          break;  // Exit the search only after at least one peak has
          // been found, even if it is necessary to go beyond
          // the nominal maximum period.
        }
      }  // end if this period is >= minimum search period.
    }  // end for each next pulse in the global period-search range.
    // Install the unvoiced candidate for this pulse.
    if (next_cands_created) {  // Register the unvoiced hyp iff there
      // was at least one voiced hyp.
      resid_peaks_[peak].future.push_back(uv_cand);
      resid_peaks_[uv_cand->end_peak].past.push_back(uv_cand);
      total_cands++;
    } else {
      delete uv_cand;
    }
    // Now all residual-pulse period hyps that start at the current
    // pulse have been generated.

    // If this pulse is one of the first few in the residual that had
    // no possible preceeding periods, mark it as an origin.
    if (resid_peaks_[peak].past.size() == 0) {  // Is this pulse an origin?
      for (size_t pp = 0; pp < resid_peaks_[peak].future.size(); ++pp) {
        resid_peaks_[peak].future[pp]->cost_sum =
            resid_peaks_[peak].future[pp]->local_cost;
        resid_peaks_[peak].future[pp]->best_prev_cand = -1;
      }
    } else {  // There are previous period hyps to consider...
      // Check if at least one UV hyp is included in the period hyps
      // that end on this peak.  If there are none, generate one by
      // cloning the best voiced hyp in the collection, but score it
      // as unvoiced.
      int32_t uv_hyps_found = 0;
      float lowest_cost =  resid_peaks_[peak].past[0]->local_cost;
      size_t lowest_index = 0;
      for (size_t pcand = 0; pcand < resid_peaks_[peak].past.size(); ++pcand) {
        if (!resid_peaks_[peak].past[pcand]->voiced) {
          uv_hyps_found++;
        } else {
          if (resid_peaks_[peak].past[pcand]->local_cost < lowest_cost) {
            lowest_index = pcand;
            lowest_cost = resid_peaks_[peak].past[pcand]->local_cost;
          }
        }
      }
      if (!uv_hyps_found) {  // clone an UV hyp from the best V hyp found.
        size_t start_peak = resid_peaks_[peak].past[lowest_index]->start_peak;
        EpochCand* uv_cand = new EpochCand;
        uv_cand->voiced = false;
        uv_cand->start_peak = start_peak;
        uv_cand->end_peak = peak;
        uv_cand->period =  resid_peaks_[peak].past[lowest_index]->period;
        uv_cand->closest_nccf_period =
            resid_peaks_[peak].past[lowest_index]->closest_nccf_period;
        uv_cand->cost_sum = 0.0;
        uv_cand->local_cost = 0.0;
        uv_cand->best_prev_cand = -1;
        float llevel_cost = level_wt_ *
            prob_voiced_[resid_peaks_[start_peak].frame_index];
        int32_t lcc_index = uv_cand->period - first_nccf_lag_;
        float lcc_value = 0.0;
        // If this period is in the normal search range, retrieve the
        // actual NCCF value for that lag.
        if ((lcc_index >= 0) && (lcc_index < n_nccf_lags_)) {
          lcc_value = resid_peaks_[start_peak].nccf[lcc_index];
        } else {
          int32_t peak_cc_index = uv_cand->closest_nccf_period - first_nccf_lag_;
          lcc_value =  resid_peaks_[start_peak].nccf[peak_cc_index];
        }
        uv_cand->local_cost = (nccf_uv_peak_wt_ * lcc_value) + llevel_cost +
            unvoiced_cost_ + reward_;
        resid_peaks_[start_peak].future.push_back(uv_cand);
        resid_peaks_[peak].past.push_back(uv_cand);
        total_cands++;
      }
    }
  }  // end of the first pass at all pulses in the residual.
  // All forward period hypotheses that start on all residual pulses
  // in the signal have now been generated, and both voiced and
  // unvoiced continuity throughout the lattice of hyps have been
  // assured.
}


void EpochTracker::DoDynamicProgramming(void) {
  // Perform the dynamic programming iterations over all pulses in
  // the residual.
  // For each pulse in the residual....
  for (size_t peak = 0; peak < resid_peaks_.size(); ++peak) {
    if (resid_peaks_[peak].past.size() == 0) {  // Is this peak an origin?
      continue;
    }
    // For each forward period hypothesis starting at this pulse...
    for (size_t fhyp = 0; fhyp < resid_peaks_[peak].future.size(); ++fhyp) {
      float min_cost = 1.0e30;  //  huge
      size_t min_index = 0;
      float forward_period =  resid_peaks_[peak].future[fhyp]->period;
      // For each of the previous period hyps ending on this pulse...
      for (size_t phyp = 0; phyp < resid_peaks_[peak].past.size(); ++phyp) {
        float sum_cost = 0.0;
        // There are 4 voicing hyps to consider: V->V  V->UV  UV->V  UV->UV
        if (resid_peaks_[peak].future[fhyp]->voiced &&
            resid_peaks_[peak].past[phyp]->voiced) {  // v->v
          float f_trans_cost = freq_trans_wt_ *
              fabs(log(forward_period / resid_peaks_[peak].past[phyp]->period));
          sum_cost = f_trans_cost + resid_peaks_[peak].past[phyp]->cost_sum;
        } else {
          if (resid_peaks_[peak].future[fhyp]->voiced &&
              !resid_peaks_[peak].past[phyp]->voiced) {  // uv->v
            float v_transition_cost = voice_transition_factor_ *
                (1.0 - voice_onset_prob_[resid_peaks_[peak].frame_index]);
            sum_cost = resid_peaks_[peak].past[phyp]->cost_sum +
                v_transition_cost;
          } else {
            if ((!resid_peaks_[peak].future[fhyp]->voiced) &&
                resid_peaks_[peak].past[phyp]->voiced) {  // v->uv
              float v_transition_cost = voice_transition_factor_ *
                  (1.0 - voice_offset_prob_[resid_peaks_[peak].frame_index]);
              sum_cost = resid_peaks_[peak].past[phyp]->cost_sum +
                  v_transition_cost;
            } else {  //  UV->UV
              sum_cost = resid_peaks_[peak].past[phyp]->cost_sum;
            }
          }
        }
        if (sum_cost < min_cost) {
          min_cost = sum_cost;
          min_index = phyp;
        }
      }  // end for each previous period hyp
      resid_peaks_[peak].future[fhyp]->cost_sum =
          resid_peaks_[peak].future[fhyp]->local_cost + min_cost;
      resid_peaks_[peak].future[fhyp]->best_prev_cand = min_index;
    }  // end for each foreward period hyp
  }  // end for each pulse in the residual signal.
  // Here ends the dynamic programming.
}


bool EpochTracker::BacktrackAndSaveOutput(void) {
  if (resid_peaks_.size() == 0) {
    fprintf(stderr, "Can't backtrack with no residual peaks\n");
    return false;
  }
  //  Now find the best period hypothesis at the end of the signal,
  //  and backtrack from there.
  float min_cost = 1.0e30;
  int32_t min_index = 0;
  // First, find a terminal peak which is the end of more than one
  // period candidate.
  size_t end = 0;
  for (size_t peak = resid_peaks_.size() - 1; peak > 0; --peak) {
    if ((resid_peaks_[peak].past.size() > 1)) {
      for (size_t ind = 0; ind < resid_peaks_[peak].past.size(); ++ind) {
        if (resid_peaks_[peak].past[ind]->cost_sum < min_cost) {
          min_cost = resid_peaks_[peak].past[ind]->cost_sum;
          min_index = ind;
        }
      }
      end = peak;
      break;
    }
  }
  if (end == 0) {
    fprintf(stderr, "No terminal peak found in DynamicProgramming\n");
    return false;
  }
  output_.clear();
  // Backtrack through the best pointers to retrieve the optimum
  // period and voicing candidates.  Save the GCI and voicing
  // estimates.
  while (1) {
    int32_t start_peak = resid_peaks_[end].past[min_index]->start_peak;
    TrackerResults tr;
    tr.resid_index = resid_peaks_[start_peak].resid_index;
    if (resid_peaks_[end].past[min_index]->voiced) {
      float nccf_period =
          resid_peaks_[end].past[min_index]->closest_nccf_period;
      // TODO(dtalkin) If the closest NCCF period is more than epsilon
      // different from the inter-pulse interval, use the inter-pulse
      // interval instead.
      tr.f0 = sample_rate_ / nccf_period;
      tr.voiced = true;
    } else {
      tr.f0 = 0.0;
      tr.voiced = false;
    }
    int32_t cc_index = resid_peaks_[end].past[min_index]->period -
        first_nccf_lag_;
    // If this period is in the normal search range, retrieve the
    // actual NCCF value for that lag.
    if ((cc_index >= 0) && (cc_index < n_nccf_lags_)) {
      tr.nccf_value = resid_peaks_[start_peak].nccf[cc_index];
    } else {
      int32_t peak_cc_index =
          resid_peaks_[end].past[min_index]->closest_nccf_period -
          first_nccf_lag_;
      tr.nccf_value =  resid_peaks_[start_peak].nccf[peak_cc_index];
    }
    output_.push_back(tr);
    size_t new_end =  resid_peaks_[end].past[min_index]->start_peak;
    min_index = resid_peaks_[end].past[min_index]->best_prev_cand;
    if (min_index < 0) {  // Has an origin pulse been reached?
      break;
    }
    end = new_end;
  }
  // NOTE:  The output_ array is in reverse time order!
  return true;
}


void EpochTracker::GetFilledEpochs(float unvoiced_pm_interval,
                                   std::vector<float>* times,
                                   std::vector<int16_t>* voicing) {
  times->clear();
  voicing->clear();
  float final_time = norm_residual_.size() / sample_rate_;
  int32_t limit = output_.size() - 1;
  int32_t i = limit;
  // Produce the output in normal time order.
  while (i >= 0) {
    int32_t i_old = i;
    float time = output_[i].resid_index / sample_rate_;
    // Note that the pulse locations of both the beginning and end
    // of any voiced period are of interest.
    if (output_[i].voiced || ((i < limit) && (output_[i+1].voiced))) {
      times->push_back(time);
      voicing->push_back(1);
      i--;
    }
    if (i == limit) {
      time = 0.0;
    }
    if ((i > 0) && (!output_[i].voiced) && (time < final_time)) {
      for ( ; i > 0; --i) {
        if (output_[i].voiced) {
          break;
        }
      }
      float next_time = final_time;
      int32_t fill_ind = 1;
      if (i > 0) {
        next_time = (output_[i].resid_index / sample_rate_) -
            (1.0 / max_f0_search_);
      }
      float now = time + (fill_ind * unvoiced_pm_interval);
      while (now < next_time) {
        times->push_back(now);
        voicing->push_back(0);
        fill_ind++;
        now = time + (fill_ind * unvoiced_pm_interval);
      }
    }
    if (i == i_old) {
      i--;
    }
  }
}


bool EpochTracker::ResampleAndReturnResults(float resample_interval,
                                            std::vector<float>* f0,
                                            std::vector<float>* correlations) {
  if ((sample_rate_ <= 0.0) || (output_.size() == 0)) {
    fprintf(stderr, 
            "Un-initialized EpochTracker or no output_ in ResampleAndReturnF0\n");
    return false;
  }
  if (resample_interval <= 0.0) {
    fprintf(stderr, "resample_interval <= 0.0 in ResampleAndReturnF0\n");
    return false;
  }
  float last_time = (output_[0].resid_index / sample_rate_) + endpoint_padding_;
  int32_t n_frames = RoundUp(last_time / resample_interval);
  f0->resize(0);
  correlations->resize(0);
  f0->insert(f0->begin(), n_frames, 0.0);
  correlations->insert(correlations->begin(), n_frames, 0.0);
  int32_t limit = output_.size() - 1;
  int32_t prev_frame = 0;
  float prev_f0 = output_[limit].f0;
  float prev_corr = output_[limit].nccf_value;
  for (int32_t i = limit; i >= 0; --i) {
    int32_t frame = RoundUp(output_[i].resid_index /
                            (sample_rate_ * resample_interval));
    (*f0)[frame] = output_[i].f0;
    (*correlations)[frame] = output_[i].nccf_value;
    if ((frame - prev_frame) > 1) {
      for (int32_t fr = prev_frame + 1; fr < frame; ++fr) {
        (*f0)[fr] = prev_f0;
        (*correlations)[fr] = prev_corr;
      }
    }
    prev_frame = frame;
    prev_corr = output_[i].nccf_value;
    prev_f0 = output_[i].f0;
  }
  for (int32_t frame = prev_frame; frame < n_frames; ++frame) {
    (*f0)[frame] = prev_f0;
    (*correlations)[frame] = prev_corr;
  }
  return true;
}


bool EpochTracker::WriteDebugData(const std::vector<float>& data,
                                  const std::string& extension) {
  if (debug_name_.empty()) {
    return true;
  }
  std::string filename = debug_name_ + "." + extension;
  if (data.size() == 0) {
    fprintf(stdout, "Data size==0 for %s in WriteDebugData\n",
               filename.c_str());
    return false;
  }
  FILE* out = fopen(filename.c_str(), "w");
  if (!out) {
    fprintf(stderr, "Can't open %s for debug output\n", filename.c_str());
    return false;
  }
  size_t  written = fwrite(&(data.front()), sizeof(data.front()),
                           data.size(), out);
  fclose(out);
  if (written != data.size()) {
    fprintf(stderr, "Problems writing debug data (%d %d)\n",
            static_cast<int>(written), static_cast<int>(data.size()));
    return false;
  }
  return true;
}

bool EpochTracker::WriteDiagnostics(const std::string& file_base) {
  if (!file_base.empty()) {
    set_debug_name(file_base);
  }
  WriteDebugData(signal_, "pcm");
  WriteDebugData(residual_, "resid");
  WriteDebugData(norm_residual_, "nresid");
  WriteDebugData(bandpassed_rms_, "bprms");
  WriteDebugData(voice_onset_prob_, "onsetp");
  WriteDebugData(voice_offset_prob_, "offsetp");
  WriteDebugData(peaks_debug_, "pvals");
  WriteDebugData(prob_voiced_, "pvoiced");
  // best_corr_ is only available after CreatePeriodLattice.
  WriteDebugData(best_corr_, "bestcorr");
  // NOTE: if WriteDiagnostics is called before the
  // DynamicProgramming, there will be nothing in output_.
  if ((!debug_name_.empty()) && (output_.size() > 2)) {
    std::string pm_name = debug_name_ + ".pmlab";
    FILE* pmfile = fopen(pm_name.c_str(), "w");
    fprintf(pmfile, "#\n");
    std::vector<float> f0;
    int32_t limit = output_.size() - 1;
    // Produce debug output in normal time order.
    for (int32_t i = limit; i >= 0; --i) {
      float time = output_[i].resid_index / sample_rate_;
      // Note that the pulse locations of both the beginning and end
      // of any voiced period are of interest.
      if (output_[i].voiced || ((i < limit) && (output_[i+1].voiced))) {
        fprintf(pmfile, "%f blue \n", time);
      } else {
        fprintf(pmfile, "%f red \n", time);
      }
      f0.push_back(time);
      f0.push_back(output_[i].f0);
      f0.push_back(output_[i].nccf_value);
    }
    fclose(pmfile);
    WriteDebugData(f0, "f0ap");
  }
  return true;
}
