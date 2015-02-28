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

#ifndef _TRACK_H_
#define _TRACK_H_

#include <stdint.h>
#include <string>
#include <vector>

#include "core/file_resource.h"
#include "core/float_matrix.h"

class Track {
 public:
  Track();
  ~Track();

  void FillFrom(const Track &track);

  // Returns the number of frames contained within this track instance.
  int num_frames() const;

  // Returns the number of channels contained within this track instance.
  int num_channels() const;

  // Returns whether the voicing is enabled on this track instance.
  bool voicing_enabled() const;

  // Sets the voicing enable flag on this track instance.
  void set_voicing_enabled(bool b);

  // Gets values for frame "f" and channel "c".
  float &a(int f, int c = 0);

  // Gets values for frame "f" and channel "c".
  float a(int f, int c = 0) const;

  // Gets voiced flags for frame "f".
  void set_v(int f, bool value);

  // Gets voiced flags for frame "f".
  bool v(int f) const;

  // Gets time instant for frame "f".
  float &t(int f);

  // Gets time instant for frame "f".
  float t(int f) const;

  // Sets the interval between frames in seconds.
  void set_shift(float shift);

  // Returns the interval between frames in seconds.
  float shift() const;

  // Modifies this track instances frame times with that given, from the
  // given frame index.
  void FillTime(float frame_shift, int start = 1);

  // Modifies this track instances frame times with the value corresponding
  // to its frame index.
  void FillTime(const Track &t);

  // Modifies the times of this track instance with the given values.
  // NOTE: the length must be equal to the number of frames in the track.
  void SetTimes(float *times, int length);

  // Modifies the voicing flag of this track instance with the given values.
bool SetVoicing(const std::vector<bool> &vuv);

  // Sets the track from the given time and value vectors.
  // All frames are set to voiced.
  template <typename ValueType>
  void Set(const std::vector<ValueType> &times, const std::vector<ValueType> &values);

  // Sets the track from the given time and value vectors.
  // All frames are set to voiced.
  template <typename ValueType>
  void Set(float time_shift, const std::vector<ValueType> &values);

  // Populates the given vector with the values for all channels for
  // the given frame index.
  void FrameOut(std::vector<float> *fv, int n) const;

  // Populates the given vector with the values for all channels for
  // the given frame index.
  void FrameOut(std::vector<double> *fv, int n) const;

  // Updates the values for all channels for the given frame index with
  // the values given.
  void FrameIn(const float &f, int n);

  // Updates the values for all channels for the given frame index with
  // the values given.
  void FrameIn(const std::vector<float> &fv, int n);

  // Updates the values for all channels for the given frame index with
  // the values given.
  void FrameIn(const std::vector<double> &fv, int n);

  // Returns the index of the frame at the given time index.
  int Index(float x) const;

  // Returns the index of the first frame below that at the given time index, x.
  int IndexBelow(float x) const;

  // Returns the index of the first frame above that at the given time index, x.
  int IndexAbove(float x) const;

  // Resizes the data and voicing flags to the given value, n, and the
  // number of channels to that given, c.
  void resize(int n, int c = 1);

  // Resets all the data associated with this track instance.
  void Clear();

  // Returns a portion of this track instance as defined by the input
  // start and end times along with the channels required.
  // Note: Ownership of the returned track is required to prevent
  // memory leaks downstream.
  Track *GetSubTrack(float start,
                     float end,
                     int ch_offset = -1,
                     int ch_size = -1) const;

  // Returns a portion of this track instance as defined by the input
  // start and end frame indices along with the channels required.
  // Note: Ownership of the returned track is required to prevent
  // memory leaks downstream.
  Track *GetSubTrack(int start_frame_index,
                     int end_frame_index,
                     int start_channel_index,
                     int end_channel_index) const;

  // Returns a std::string representation of the header of this track:
  // The number of frames in this track
  // The number of channels in this track
  // The interval between frames
  // Whether voicing is enabled or not
  std::string HeaderToString() const;

  // Returns a std::string representation of this track instance containing:
  // The number of frames in this track
  // The number of channels in this track
  // The interval between frames
  // Whether voicing is enabled or not
  // values
  // If precision is specified, float numbers are printed out up to
  // the specified number of decimal places.
  std::string ToString(uint32_t precision) const;
  std::string ToString() const;

  // Saves this track instance to the given FileResource.
  bool Save(FileResource *fr) const;

  // Saves this track instance to the given FileResource.
  bool Save(const std::string &filename, bool ascii) const;

  const FloatMatrix &data() const;

  // Resizes the track to the number of frames and channels given,
  // copying the given data into the track.
  void CopyFrom(const char *v, int32_t num_frames, int32_t num_channels);

  // Sets this track to be a combination of two input tracks
  bool SetCombinedTrack(const Track &track_a,
                        const Track &track_b);

  // Sets this track to be combination of three input tracks
  bool SetCombinedTrack(const Track &track_a,
                        const Track &track_b,
                        const Track &track_c);

  // Pads a track by repeating the last frame num_pads times.
  bool Pad(int num_pads);

  // Makes the current track and ref track equal lengths, use the longest length
  // and zero pad the shorter track if the are different lengths.
  void MakeSameSize(Track *track);

  // Makes the current track and ref track equal lengths, use the longest length
  // and zero pad the shorter tracks if any are shorter than the longest.
  void MakeSameSize(Track *track_a,
                    Track *track_b);

 private:
  // Holds the number of frames of this track instance.
  int num_frames_;

  // Holds the number of channels of this track instance.
  int num_channels_;

  // Holds the frames values and times of this track instance.
  FloatMatrix data_;

  // Holds the voicing flags of this track instance.
  std::vector<bool> val_;

  // Holds whether this track instance has voicing enabled.
  bool voicing_enabled_;

  // The frame interval in seconds.
  float shift_;
};

// Applies logarithm to all data values. Invalid values are set to INVALID_LOG.
void ConvertToLogarithmic(Track *t);

// Returns a new vector with the track data as a single sequence of
// values. The caller takes ownership fo the pointer.
float *TrackToFloatPointer(const Track &track, int *num_samples);

inline int Track::num_frames() const {
  return num_frames_;
}

inline int Track::num_channels() const {
  return num_channels_;
}

inline float &Track::a(int f, int c) {
  return data_.Get(c + 1, f);
}

inline float Track::a(int f, int c) const {
  return data_.Get(c + 1, f);
}

inline void Track::set_v(int f, bool value) {
  // If a writable reference is used, force the track into
  // voicing enabled mode.
  voicing_enabled_ = true;
  val_[f] = value;
}

inline bool Track::v(int f) const {
  return val_[f];
}

inline float &Track::t(int f) {
  return data_.Get(0, f);
}

inline float Track::t(int f) const {
  return data_.Get(0, f);
}

inline void Track::set_shift(float shift) {
  shift_ = shift;
}

inline const FloatMatrix &Track::data() const {
  return data_;
}

inline bool Track::voicing_enabled() const {
  return voicing_enabled_;
}

inline void Track::set_voicing_enabled(bool b) {
  voicing_enabled_ = b;
}

inline bool Track::Save(const std::string &filename, bool ascii) const {
  FileResource fr(filename, "wb");
  if (!fr.Get()) {
    fprintf(stderr, "Failed to write '%s'", filename.c_str());
    return false;
  }
  if (!ascii) {
    return Save(&fr);
  }
  const std::string data = ToString();
  if (fprintf(fr.fp(), "%s", data.c_str()) != data.size()) {
    return false;
  }
  return true;
}

template <typename ValueType>
void Track::Set(const std::vector<ValueType> &times,
                const std::vector<ValueType> &values) {
  if (times.size() != values.size()) {
    fprintf(stderr, "Length of time and value vectors should equal (%d != %d)",
                times.size(), values.size());
    return;
  }
  resize(times.size());
  for (int n = 0; n < times.size(); ++n) {
    t(n) = times[n];
    a(n) = values[n];
    set_v(n, true);
  }
}

template <typename ValueType>
void Track::Set(float time_shift, const std::vector<ValueType> &values) {
  resize(values.size());
  FillTime(time_shift, 0);
  for (int n = 0; n < values.size(); ++n) {
    a(n) = values[n];
    set_v(n, true);
  }
}

#endif  // _TRACK_H_
