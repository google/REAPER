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

// EpochTracker estimates the location of glottal closure instants
// (GCI), also known as "epochs" from digitized acoustic speech
// signals.  It simultaneously estimates the local fundamental
// frequency (F0) and voicing state of the speech on a per-epoch
// basis.  Various output methods are available for retrieving the
// results.
//
// The processing stages are:
//  * Optionally highpass the signal at 80 Hz to remove rumble, etc.
//  * Compute the LPC residual, obtaining an approximation of the
//    differentiated glottal flow.
//  * Normalize the amplitude of the residual by a local RMS measure.
//  * Pick the prominent peaks in the glottal flow, and grade them by
//    peakiness, skew and relative amplitude.
//  * Compute correlates of voicing to serve as pseudo-probabilities
//    of voicing, voicing onset and voicing offset.
//  * For every peak selected from the residual, compute a normalized
//    cross-correlation function (NCCF) of the LPC residual with a
//    relatively short reference window centered on the peak.
//  * For each peak in the residual, hypothesize all following peaks
//    within a specified F0 seaqrch range, that might be the end of a
//    period starting on that peak.
//    * Grade each of these hypothesized periods on local measures of
//      "voicedness" using the NCCF and the pseudo-probability of voicing
//      feature.
//    * Generate an unvoiced hypothesis for each period and grade it
//      for "voicelessness".
//    * Do a dynamic programming iteration to grade the goodness of
//      continuity between all hypotheses that start on a peak and
//      those that end on the same peak.  For voiced-voiced
//      connections add a cost for F0 transitions.  For
//      unvoiced-voiced and voiced-unvoiced transitions add a cost
//      that is modulated by the voicing onset or offset inverse
//      pseudo-probability.  Unvoiced-unvoiced transitions incur no cost.
//  * Backtrack through the lowest-cost path developed during the
//    dynamic-programming stage to determine the best peak collection
//    in the residual.  At each voiced peak, find the peak in the NCCF
//    (computed above) that corresponds to the duration closest to the
//    inter-peak interval, and use that as the inverse F0 for the
//    peak.
//
// A typical calling sequence might look like:
/* ==============================================================
   EpochTracker et;
   et.Init();  // Prepare the instance for, possibly, multiple calls.
   Track* f0;  // for returning the F0 track
   Track* pm;  // for returning the epoch track
   if (!et.ComputeEpochs(my_input_waveform, &pm, &f0)) {
     exit(-2);  // problems in the epoch computations
   }
   DoSomethingWithTracks(f0, pm);
   delete f0;
   delete pm;
   ============================================================== */
//
// NOTE: Any client of this code inherits the Google command-line flags
// defined in epoch_tracker.cc.  These flags are processed in the Init()
// method, and override both default and params-sourced settings.
//
// As currently written, this is a batch process.  Very little has
// been done to conserve either memory or CPU.  The aim was simply to
// make the best possible tracker.  As will be seen in the
// implementation, there are many parameters that can be adjusted to
// influence the processing.  It is very unlikely that the best
// parameter setting is currently expressed in the code!  However, the
// performance, as written, appears to be quite good on a variety of voices.

#ifndef _EPOCH_TRACKER_H_
#define _EPOCH_TRACKER_H_

#include <memory>
#include <stdint.h>
#include <vector>
#include <string>

static const float kExternalFrameInterval = 0.005;
static const float kInternalFrameInterval = 0.002;
static const float kMinF0Search = 40.0;
static const float kMaxF0Search = 500.0;
static const float kUnvoicedPulseInterval = 0.01;
static const float kUnvoicedCost = 0.9;
static const bool kDoHighpass = true;
static const bool kDoHilbertTransform = false;
static const char kDebugName[] = "";


class EpochTracker {
 public:
  EpochTracker(void);

  virtual ~EpochTracker(void);

  // Set the default operating parameters of the tracker.
  void SetParameters(void);

  // NOTE: The following methods are exposed primarily for algorithm
  // development purposes, where EpochTracker is used in a developer's test
  // harness.  These need not/should not be called directly in normal use.

  // Prepare the instance for use.  Some sanity check is made on the
  // parameters, and the instance is reset so it can be reused
  // multiple times by simply calling Init() for each new input
  // signal. frame_interval determines the framing for some of the
  // internal feature computations, and for the periodic resampling of
  // F0 that will occur during final tracking result output.  min_
  // and max_f0_search are the bounding values, in Hz, for the F0
  // search.
  // NOTE: This Init method is DEPRECATED, and is only retained to
  // support legacy code.  IT MAY GO AWAY SOON.  This is NOT to be
  // used with ComputeEpochs().
  bool Init(const int16_t* input, int32_t n_input, float sample_rate,
            float min_f0_search, float max_f0_search,
            bool do_highpass, bool do_hilbert_transform);

  // Set the name for various intermediate features and other signals
  // that may be written to files used during debug and development of
  // the algorithm.  If this is set to the empty std::string, no debug
  // signals will be output.
  void set_debug_name(const std::string& debug_name) {
    debug_name_ = debug_name;
  }

  std::string debug_name(void) { return debug_name_; }

  // Compute the Hilbert transform of the signal in input, and place
  // the floating-point results in output.  output must be at least
  // n_input samples long.  TODO(dtalkin): Make these vector inputs
  // and outputs.
  void HilbertTransform(int16_t* input, int32_t n_input, float* output);

  // Apply a highpass filter to the signal in input.  The filter
  // corner frequency is corner_freq, and the duration, in seconds, of
  // the Hann-truncated symmetric FIR is in fir_duration.  The
  // transition bandwidth is the inverse of fir_duration.  The return
  // value is a pointer to the filtered result, which is the same
  // length as the input (n_input).  It is up to the caller to free
  // this returned memory.  TODO(dtalkin): Make this vector I/O and
  // supply the output as floats.
  int16_t* HighpassFilter(int16_t* input, int32_t n_input,
                          float sample_rate, float corner_freq,
                          float fir_duration);

  // Compute the LPC residual of the speech signal in input.
  // sample_rate is the rate of both the input and the residual to be
  // placed in output.  The order of the LPC analysis is automatically
  // set to be appropriate for the sample rate, and the output is
  // integrated so it approximates the derivative of the glottal flow.
  bool GetLpcResidual(const std::vector<float>& input, float sample_rate,
                      std::vector<float>* output);

  // Compute the normalized cross-correlation function (NCCF) of the
  // signal in data, starting at sample start.  size is the number of
  // samples to include in the inner product. Compute n_lags
  // contiguous correlations starting at a delay of first_lag samples.
  // Return the resulting n_lags correlation values in corr.  Note
  // that the NCCF is bounded by +-1.0.
  void CrossCorrelation(const std::vector<float>& data, int32_t start,
                        int32_t first_lag, int32_t n_lags,
                        int32_t size, std::vector<float>* corr);

  // Compute the band-limited RMS of the signal in input, which is
  // sampled at sample_rate.  low_limit and high_limit are the
  // frequency bounds, in Hz, within which th RMS is measured.
  // frame_interval is the period of the RMS signal returned in
  // output_rms.  frame_dur is the duration, in seconds, of the
  // Hanning window used for each measurement.
  bool GetBandpassedRmsSignal(const std::vector<float>& input, float sample_rate,
                              float low_limit, float high_limit, float frame_interval,
                              float frame_dur,  std::vector<float>* output_rms);

  // Compute the RMS of positive and negative signal values separately.
  // The signal is made to be zero mean before this computation.  Any
  // imbalance in these measures is an indication of asymmetrical peak
  // distribution, which is charactristic of the LPC residual of voiced speech.
  void GetSymmetryStats(const std::vector<float>& data, float* pos_rms,
                        float* neg_rms, float* mean);

  // Normalize the input signal based on a local measure of its RMS.
  void NormalizeAmplitude(const std::vector<float>& input, float sample_rate,
                          std::vector<float>* output);

  // Apply a Hann weighting to the signal in input starting at
  // sample index offset.  The window will contain size samples, and
  // the windowed signal is placed in output.
  void Window(const std::vector<float>& input, int32_t offset, size_t size,
              float* output);

  // Computes signal polarity (-1 for negative, +1 for
  // positive). Requires data to be initialized via Init(...). Returns
  // false if there's an error.
  bool ComputePolarity(int *polarity);

  // Compute NCCF, NCCF peak locations and values, bandpass RMS,
  // residual, symmetry statistics (and invert residual, if necessary),
  // normalized residual, residual peaks and values.  Finally, generate
  // the pulse working array in preparation for dynamic programming.
  bool ComputeFeatures(void);

  // Write all of data to a file, wht name of which is
  // debug_name_ _ "." + extension.  If debug_name_ is empty, do nothing.
  bool WriteDebugData(const std::vector<float>& data,
                      const std::string& extension);

  // Write a collection of debugging signals to separate files with
  // various, internally-defined name extensions.  If file_base is not
  // empty, use this as the base path for all of the files.  If file
  // base is empty, use debug_name_ as the base path.  If both are
  // empty, do nothing.
  bool WriteDiagnostics(const std::string& file_base);

  // After Init, ComputeFeatures and CreatePeriodLattice have been
  // successfully called, TrackEpochs should be called to do the
  // actual tracking of epochs (GCI) and to estimate the corresponding
  // F0.  This method integrates the information from all of the
  // features, including the LPC residual peaks and the NCCF values,
  // to find the optimum period assignments and voicing state
  // assignments over the entire signal.  The results are left in
  // internal storage, pending retrieval by other methods.
  bool TrackEpochs(void);

  // Create a lattice of glottal period hypotheses in preparation for
  // dynamic programming.  This fills out most of the data fields in
  // resid_peaks_. This must be called after ComputeFeatures.
  void CreatePeriodLattice(void);

  // Apply the Viterbi dynamic programming algorithm to find the best
  // path through the period hypothesis lattice created by
  // CreatePeriodLattice.  The backpointers and cumulative scores are
  // left in the relevant fields in resid_peaks_.
  void DoDynamicProgramming(void);

  // Backtrack through the best pointers in the period hypothesis
  // lattice created by CreatePeriodLattice and processed by
  // DoDynamicProgramming.  The estimated GCI locations
  // (epochs) and the corresponding F0 and voicing-states are placed
  // in the output_ array pending retrieval using other methods.
  bool BacktrackAndSaveOutput(void);

  // Resample the per-period F0 and correlation data that results from
  // the tracker to a periodic signal at an interval of
  // resample_interval seconds.  Samples returned are those nearest in
  // time to an epoch.  Thus, if the resample_interval is greater than
  // the local epoch interval, some epochs, and their period
  // information, will be skipped.  Conversely, if the
  // resample_interval is less than the local epoch interval,
  // measurements will be replicated as required.
  bool ResampleAndReturnResults(float resample_interval,
                                std::vector<float>* f0,
                                std::vector<float>* correlations);

  // Convert the raw backtracking results in output_ into
  // normal-time-order epoch markers.  In unvoiced regions, fill with
  // regularly-spaced pulses separated by unvoiced_pm_interval
  // seconds. The epoch/pulse times are returned in times re the
  // utterance beginning, and the corresponding voicing states in
  // voicing (0=unvoiced; 1=voiced).  This can only be called after
  // TrackEpochs.
  void GetFilledEpochs(float unvoiced_pm_interval, std::vector<float>* times,
                       std::vector<int16_t>* voicing);

  // Setters.
  void set_do_hilbert_transform(bool v) { do_hilbert_transform_ = v; }
  void set_do_highpass(bool v) { do_highpass_ = v; }
  void set_external_frame_interval(float v) { external_frame_interval_ = v; }
  void set_unvoiced_pulse_interval(float v) { unvoiced_pulse_interval_ = v; }
  void set_min_f0_search(float v) { min_f0_search_ = v; }
  void set_max_f0_search(float v) { max_f0_search_ = v; }
  void set_unvoiced_cost(float v) { unvoiced_cost_ = v; }

 private:
  // Search the signal in norm_residual_ for prominent negative peaks.
  // Grade the peaks on a combination of amplitude, "peakiness" and
  // skew. (It is expected that the glottal pulses will
  // have a relatively slow fall, and a rapid rise.)  Place the
  // selected and graded pulses in resid_peaks_.
  void GetResidualPulses(void);

  // Create pseudo-probability functions in voice_onset_prob_ and
  // voice_offset_prob_ that attempt to indicate the time-varying
  // probability that a voice onset or offset is occurring.
  // Presently, this is based solely on the derivative of the
  // bandpassed RMS signal, bandpassed_rms_.
  void GetVoiceTransitionFeatures(void);

  // Generate a pseudo-probability function that attempts to corespond
  // to the probability that voicing is occurring.  This is presently
  // based solely on the bandpassed RMS signal, bandpassed_rms_.
  void GetRmsVoicingModulator(void);

  // Free memory, and prepare the instance for a new signal.
  void CleanUp(void);

  // Scan the signal in input searching for all local maxima that
  // exceed thresh.  The indices corresponding to the location of the
  // peaks are placed in output.  The first entry in output is always
  // the location of the largest maximum found.
  int32_t FindNccfPeaks(const std::vector<float>& input, float thresh,
                        std::vector<int16_t>* output);

  // Compute the NCCF with the reference window centered on each of
  // the residual pulses identified in GetResidualPulses.  window_dur
  // is the duration in seconds of the correlation inner product.
  // After the NCCF for each residual pulse is computed, it is
  // searched for local maxima that exceed peak_thresh.  These peak
  // locations and the full NCCF are saved in the corresponding
  // elements of the resid_peaks_ array of structures.
  void GetPulseCorrelations(float window_dur, float peak_thresh);


 private:
  // EpochCand stores all of the period hypotheses that can be
  // generated from the peaks found in the LPC residual.  It also
  // maintains the cumulative path costs and backpointers generated
  // during dynamic programming.
  struct EpochCand {
    int32_t period;  // # of samples in this period candidate
    float local_cost;  // cost of calling this a period (or unvoiced)
    float cost_sum;  // cumulative cost from DP
    int32_t start_peak;  // index in resid_peaks_ where this period hyp starts
    int32_t end_peak;  // where this period ends
    int32_t best_prev_cand;  // backpointer used after DP
    int32_t closest_nccf_period;  // per. implied by the closest correlation peak
    bool voiced;  // hypothesized voicing state for this cand.
  };

  typedef std::vector<EpochCand*> CandList;

  // The ResidPeak stores data for each residual impulse.  The array
  // of these in resid_peaks_ serves as input to the dynamic
  // programming search for GCI, voicing state and F0.
  struct ResidPeak {
    int32_t resid_index;  // index into the resid_ array of this peak
    int32_t frame_index;  // index into the feature arrays for this peak
    float peak_quality;  // "goodness" measure for this peak
    std::vector<float> nccf;  // the NCCF computed centered on this peak
    std::vector<int16_t> nccf_periods;  // periods implied by major peaks in nccf
    CandList future;  // period candidates that start on this peak
    CandList past;  // period candidates that end on this peak
  };

  struct TrackerResults {
    bool voiced;
    float f0;
    int32_t resid_index;
    float nccf_value;
  };
  typedef std::vector<TrackerResults> TrackerOutput;

 protected:
  std::vector<ResidPeak> resid_peaks_;  // array of structures used to
  // store the peak search lattice
  TrackerOutput output_;  // Array of time stamped results of the tracker.
  // signal_, residual_, norm_residual and peaks_debug_ are all
  // sampled at the original signal input sample_rate_.
  std::vector<float> signal_;  // floating version of input speech signal
  std::vector<float> residual_;  // LPC residual normalized for constant DC.
  std::vector<float> norm_residual_;  // LPC residual normalized by its local RMS.
  std::vector<float> peaks_debug_;  // for debug output of residual peak candidates
  // bandpassed_rms_, voice_onset_prob_, voice_offset_prob_ and
  // prob_voiced_ are all sampled with a period of internal_frame_interval_.
  std::vector<float> bandpassed_rms_;  // RMS sampled at internal_frame_interval_
  std::vector<float> voice_onset_prob_;  // prob that a voice onset is occurring
  std::vector<float> voice_offset_prob_;  // prob that a voice offset is occurring
  std::vector<float> prob_voiced_;  // prob that voicing is occurring
  std::vector<float> best_corr_;  // An array of best NCCF vals for all resid peaks.
  std::vector<float> window_;  // Hann weighting array for Window()
  float sample_rate_;  // original input signal sample rate in Hz

  float positive_rms_;  // RMS of all positive, non-zero samples in residual_
  float negative_rms_;  // RMS of all negative, non-zero samples in residual_
  int32_t n_feature_frames_;  // The number of feature frames available
  // for all features computed at
  // internal_frame_interval_.
  int32_t first_nccf_lag_;  // The index of the first correlation of the
  // NCCF.  This is determined by
  // max_f0_search_.
  int32_t n_nccf_lags_;  // The number of correlations computed at each
  // residual peak candidate.  This is determined
  // by max_f0_search_ and min_f0_search_.
  std::string debug_name_;  // The base path name for all debug output files.

  // Below are all of the parameters that control the functioning of
  // the tracker.  These are all set to default known-to-work values in
  // SetParameters().

  // Control parameters available to clients of EpochTracker.
  float external_frame_interval_;  // Frame interval for final output of F0.
  float unvoiced_pulse_interval_;  // Pulse interval in unvoiced regions
  float min_f0_search_;  // minimum F0 to search for (Hz)
  float max_f0_search_;  // maximum F0 to search for (Hz)
  bool do_highpass_;  // Highpass input sighal iff true.
  bool do_hilbert_transform_;  // Hilbert trans. input data iff true.

  // Internal feature-computation Parameters:
  float internal_frame_interval_;  // interval, in seconds, between frame onsets
  // for the high-pass filter
  float corner_frequency_;
  float filter_duration_;
  // for the LPC inverse filter.
  float frame_duration_;  // window size (sec)
  float lpc_frame_interval_;  // (sec)
  float preemphasis_;  // preemphasis for LPC analysis
  float noise_floor_;  // SNR in dB simulated during LPC analysis.
  // for computing LPC residual peak quality.
  float peak_delay_;  // for measuring prominence
  float skew_delay_;  // for measuring shape
  float peak_val_wt_;
  float peak_prominence_wt_;
  float peak_skew_wt_;
  float peak_quality_floor_;
  // for computing voice-transition pseudo-probabilities
  float time_span_;  // the interval (sec) centered on the
  // measurement point, used to
  // compute parameter deltas
  float level_change_den_;  // max. dB level change
  // expected over time_span_ for
  // bandpassed RMS
  // for computing pseudo-probability of voicing
  float min_rms_db_;  // level floor in dB
  // window size for computing amplitude-normalizing RMS
  float ref_dur_;
  // low and high frequency limits for bandpassed RMS used in voicing indicator
  float min_freq_for_rms_;
  float max_freq_for_rms_;
  // duration of integrator for bandpassed RMS
  float rms_window_dur_;
  // window duration, in seconds, for NCCF computations
  float correlation_dur_;
  // ignore any NCCF peaks less than this
  float correlation_thresh_;

  // Parametrs used by the dynamic-programming tracker:
  // reward for inserting another period
  float reward_;
  // weight given to deviation of inter-pulse interval from the
  // closest NCCF peak lag
  float period_deviation_wt_;
  // weight given to the quality of the residual peak
  float peak_quality_wt_;
  // cost of the unvoiced hypothesis
  float unvoiced_cost_;
  // cost of high NCCF values in hypothetical unvoiced regions
  float nccf_uv_peak_wt_;
  // weight given to period length
  float period_wt_;
  // weight given to the pseudo-probability of voicing feature
  float level_wt_;
  // weight given to period-length differences between adjacent periods.
  float freq_trans_wt_;
  // cost of switching between voicing states; modulated by voicing
  // onset/offset probs.
  float voice_transition_factor_;

  // Parameters used to generate final outputs:
  // pad time in seconds to add to the last measured period during
  // output of periodically-resampled data
  float endpoint_padding_;
};


#endif  // _EPOCH_TRACKER_H_
