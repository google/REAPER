# REAPER: Robust Epoch And Pitch EstimatoR

This is a speech processing system.  The _reaper_ program uses the
EpochTracker class to simultaneously estimate the location of
voiced-speech "epochs" or glottal closure instants (GCI), voicing
state (voiced or unvoiced) and fundamental frequency (F0 or "pitch").
We define the local (instantaneous) F0 as the inverse of the time
between successive GCI.

This code was developed by David Talkin at Google. This is not an
official Google product (experimental or otherwise), it is just
code that happens to be owned by Google.

## Downloading and Building _reaper_
```
cd convenient_place_for_repository
git clone https://github.com/google/REAPER.git
cd REAPER
mkdir build   # In the REAPER top-level directory
cd build
cmake ..
make
```

_reaper_ will now be in `convenient_place_for_repository/REAPER/build/reaper`

You may want to add that path to your PATH environment variable or
move _reaper_ to your favorite bin repository.

Example:

To compute F0 (pitch) and pitchmark (GCI) tracks and write them out as ASCII files:

`reaper -i /tmp/bla.wav -f /tmp/bla.f0 -p /tmp/bla.pm -a`


## Input Signals:

As written, the input stage expects 16-bit, signed integer samples.
Any reasonable sample rate may be used, but rates below 16 kHz will
introduce increasingly coarse quantization of the results, and higher
rates will incur quadratic increase in computational requirements
without gaining much in output accuracy.

While REAPER is fairly robust to recording quality, it is designed for
use with studio-quality speech signals, such as those recorded for
concatenation text-to-speech systems.  Phase distortion, such as that
introduced by some close-talking microphones or by well-intended
recording-studio filtering, including rumble removal, should be
avoided, for best results.  A rumble filter is provided within REAPER
as the recommended (default) high-pass pre-filtering option, and is
implemented as a symmetric FIR filter that introduces no phase
distortion.

The help text _(-h)_ provided by the _reaper_ program describes
various output options, including debug output of some of the feature
signals.  Of special interest is the residual waveform which may be
used to check for the expected waveshape.  (The residual has a
_.resid_ filename extension.) During non-nasalized, open vocal tract
vocalizations (such as /a/), each period should show a somewhat noisy
version of the derivative of the idealized glottal flow.  If the computed
residual deviates radically from this ideal, the Hilbert transform
option _(-t)_ might improve matters.

## The REAPER Algorithm:

The process can be broken down into the following phases:
* Signal Conditioning
* Feature Extraction
* Lattice Generation
* Dynamic Programming
* Backtrace and Output Generation


## Signal Conditioning

DC bias and low-frequency noise are removed by high-pass filtering,
and the signal is converted to floating point.  If the input is known
to have phase distortion that is impacting tracker performance, a
Hilbert transform, optionally done at this point, may improve
performance.


## Feature Extraction

The following feature signals are derived from the conditioned input:
* Linear Prediction residual:
  This is computed using the autocorrelation method and continuous
  interpolation of the filter coefficients.  It is checked for the
  expected polarity (negative impulses), and inverted, if necessary.
* Amplitude-normalized prediction residual:
  The normalization factor is based on the running, local RMS.
* Pseudo-probability of voicing:
  This is based on a local measure of low-frequency energy normalized
  by the peak energy in the utterance.
* Pseudo-probability of voicing onset:
  Based on a forward delta of lowpassed energy.
* Pseudo-probability of voicing offset:
  Based on a backward delta of lowpassed energy.
* Graded GCI candidates:
  Each negative peak in the normalized residual is compared with the
  local RMS.  Peaks exceeding a threshold are selected as GCI candidates,
  and then graded by a weighted combination of peak amplitude, skewness,
  and sharpness. Each of the resulting candidates is associated with the
  other feature values that occur closest in time to the candidate.
* Normalized cross-correlation functions (NCCF) for each GCI candidate:
  The correlations are computed on a weighted combination of the speech
  signal and its LP residual.  The correlation reference window for
  each GCI candidate impulse is centered on the inpulse, and
  correlations are computed for all lags in the expected pitch period range.


## Lattice Generation

Each GCI candidate (pulse) is set into a lattice structure that links
preceding and following pulses that occur within minimum and maximum
pitch period limits that are being considered for the utterance.
These links establish all of the period hypotheses that will be
considered for the pulse.  Each hypothesis is scored on "local"
evidence derived from the NCCF and peak quality measures.  Each pulse
is also assigned an unvoiced hypothesis, which is also given a score
based on the available local evidence.  The lattice is checked, and
modified, if necessary to ensure that each pulse has at least one
voiced and one unvoiced hypothesis preceding and following it, to
maintain continuity for the dynamic programming to follow.
(Note that the "scores" are used as costs during dynamic programming,
so that low scores encourage selection of hypotheses.)


## Dynamic Programming

```
For each pulse in the utterance:
  For each period hypotheses following the pulse:
    For each period hypothesis preceding the pulse:
      Score the transition cost of connecting the periods.  Choose the
      minimum overall cost (cumulative+local+transition) preceding
      period hypothesis, and save its cost and a backpointer to it.
      The costs of making a voicing state change are modulated by the
      probability of voicing onset and offset.  The cost of
      voiced-to-voiced transition is based on the delta F0 that
      occurs, and the cost of staying in the unvoiced state is a
      constant system parameter.
```

## Backtrace and Output Generation

Starting at the last peak in the utterance, the lowest cost period
candidate ending on that peak is found.  This is the starting point
for backtracking.  The backpointers to the best preceding period
candidates are then followed backwards through the utterance.  As each
"best candidate" is found, the time location of the terminal peak is
recorded, along with the F0 corresponding to the period, or 0.0 if the
candidate is unvoiced.  Instead of simply taking the inverse of the
period between GCI estimates as F0, the system refers back to the NCCF
for that GCI, and takes the location of the NCCF maximum closest to
the GCI-based period as the actual period.  The output array of F0 and
estimated GCI location is then time-reversed for final output.

