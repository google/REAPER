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
#include "core/track.h"

#include <math.h>
#include <memory>
#include <stdlib.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#if !defined(INVALID_LOG)
#define INVALID_LOG -1.0E+10F
#endif  // INVALID_LOG

namespace {

int ToInt(const std::string &s) {
  char *p = 0;
  int32_t r = strtol(s.c_str(), &p, 10);
  return static_cast<int>(r);
}

std::string ToString(float n, int p) {
  char buf[1000];
  buf[0] = 0;
  snprintf(buf, sizeof(buf), "%.*f", p, n);
  return buf;
}

std::string ToString(uint32_t n) {
  char buf[1000];
  buf[0] = 0;
  snprintf(buf, sizeof(buf), "%" PRIu32, n);
  return buf;
}

double ToDouble(const std::string &s) {
  char *p = 0;
  return strtod(s.c_str(), &p);
}

float ToFloat(const std::string &s) {
  return static_cast<float>(ToDouble(s));
}

//
// Default size of character I/O buffers:
//
const uint32_t kMaxCharBufSize = 8192;  // 8k

std::string GetToken(FileResource *fr) {
  char buff[kMaxCharBufSize];
  uint32_t i = 0;
  bool foundData = false;
  do {
    buff[i] = fgetc(fr->fp());
    if ((iscntrl((unsigned char)buff[i]) || isspace((unsigned char)buff[i])) &&
        !foundData) {
      continue;
    }
    foundData = true;
    if (iscntrl((unsigned char)buff[i]) || isspace((unsigned char)buff[i])) {
      break;
    }
    i++;
  } while (i < kMaxCharBufSize && !fr->eof());
  buff[i] = 0;
  return std::string(buff);
}

// Find the first whitespace/newline delimited token in the character
// array c.  Return the stripped token in s.  return the position of
// the first character following the token in tok_end.  If the end of
// c is found during the search, return true, else return false. If no
// token is found, return 0 in tok_end.
bool GetTokenFromChars(const char *c, std::string *s, int32_t *tok_end) {
  int32_t i = 0;
  s->clear();
  *tok_end = 0;
  // skip initial whitespace
  while (c[i] && ((c[i] == ' ') || (c[i] == '\t') || (c[i] == '\n'))) {
    i++;
  }
  if (!c[i]) {
    return true;
  }
  int32_t start = i;
  while (c[i] && (c[i] != ' ') && (c[i] != '\t') && (c[i] != '\n')) {
    i++;
  }
  *tok_end = i;

  if (c[i-1] == '\r') {
    --i;
  }
  s->assign(c + start, i - start);
  return c[i] == 0;
}

}  // namespace


Track::Track() {
  num_frames_ = 0;
  num_channels_ = 0;
  shift_ = 0.0F;
  voicing_enabled_ = false;
}

void Track::FillFrom(const Track &t) {
  num_frames_ = t.num_frames_;
  num_channels_ = t.num_channels_;
  shift_ = t.shift_;
  data_ = t.data_;
  val_ = t.val_;
  voicing_enabled_ = t.voicing_enabled_;
}


Track::~Track() {
  Clear();
}

bool Track::Save(FileResource *fr) const {
  fprintf(fr->fp(), "EST_File Track\n");
  fprintf(fr->fp(), "DataType binary2\n");
  fprintf(fr->fp(), "NumFrames %d\n", num_frames());
  fprintf(fr->fp(), "NumChannels %d\n", num_channels());
  fprintf(fr->fp(), "FrameShift %f\n", shift());
  if (voicing_enabled_) {
    fprintf(fr->fp(), "VoicingEnabled true\n");
  } else {
    fprintf(fr->fp(), "VoicingEnabled false\n");
  }
  fprintf(fr->fp(), "EST_Header_End\n");

  // write time information
  for (int i = 0; i < num_frames(); ++i) {
    float f = t(i);
    if (fwrite(&f, sizeof(f), 1, fr->fp()) != 1) {
      fprintf(stderr, "Failed to write to track file");
      return false;
    }
  }

  // write voicing information
  if (voicing_enabled_) {
    for (int i = 0; i < num_frames(); ++i) {
      char vu = v(i) ? 1 : 0;
      if (fwrite(&vu, sizeof(vu), 1, fr->fp()) != 1) {
        fprintf(stderr, "Failed to write to track file");
        return false;
      }
    }
  }

  // write coefficient information
  for (int i = 0; i < num_frames(); ++i) {
    for (int j = 0; j < num_channels(); ++j) {
      float f = a(i, j);
      if (fwrite(&f, sizeof(f), 1, fr->fp()) != 1) {
        fprintf(stderr, "Failed to write to track file");
        return false;
      }
    }
  }
  return true;
}

int Track::Index(float x) const {
  if (num_frames() > 1) {  // if single frame, return that index (0)
    int bst, bmid, bend;
    bst = 1;
    bend = num_frames();
    if (x < t(bst)) {
      bmid = bst;
    }
    if (x >= t(bend - 1)) {
      bmid = bend - 1;
    } else {
      while (1) {
        bmid = bst + (bend - bst) / 2;
        if (bst == bmid) {
          break;
        } else if (x < t(bmid)) {
          if (x >= t(bmid - 1)) {
            break;
          }
          bend = bmid;
        } else {
          bst = bmid;
        }
      }
    }
    if (fabs(x - t(bmid)) < fabs(x - t(bmid - 1))) {
      return bmid;
    } else {
      return bmid - 1;
    }
  }
  return num_frames() - 1;
}

int Track::IndexBelow(float x) const {
  for (int i = 1; i < num_frames(); ++i) {
    if (x <= t(i)) {
      return i - 1;
    }
  }
  return num_frames() - 1;
}

int Track::IndexAbove(float x) const {
  for (int i = 0; i < num_frames(); ++i) {
    if (x <= t(i)) {
      return i;
    }
  }
  return num_frames() - 1;
}

std::string Track::HeaderToString() const {
  std::string ret;
  ret = "EST_File Track\n";
  ret += "DataType ascii\n";
  ret += "NumFrames " + ::ToString(num_frames()) + "\n";
  ret += "NumChannels " + ::ToString(num_channels()) + "\n";
  ret += "FrameShift " + ::ToString(shift(), 5) + "\n";
  ret += "VoicingEnabled ";
  if (voicing_enabled_) {
    ret += "true\n";
  } else {
    ret += "false\n";
  }
  ret += "EST_Header_End\n";
  return ret;
}

std::string Track::ToString(uint32_t precision) const {
  std::string ret;
  ret = HeaderToString();
  char buf[1024];
  for (int i = 0; i < num_frames(); ++i) {
    snprintf(buf, 1024, "%f %d", t(i), v(i) ? 1 : 0);
    ret += buf;
    for (int j = 0; j < num_channels(); ++j) {
      ret += " ";
      ret += ::ToString(a(i, j), precision);
    }
    ret += std::string("\n");
  }
  return ret;
}

std::string Track::ToString() const {
  const uint32_t kPrintOutPrecision = 6;
  return ToString(kPrintOutPrecision);
}

float Track::shift() const {
  return shift_;
}

void Track::FillTime(float frame_shift, int start) {
  shift_ = frame_shift;
  for (int i = 0; i < num_frames_; ++i) {
    data_.Set(0, i, frame_shift * (i + start));
  }
}

void Track::FillTime(const Track &t) {
  for (int i = 0; i < num_frames_; ++i) {
    data_.Set(0, i, t.t(i));
  }
}

void Track::SetTimes(float *times, int length) {
  if (length != num_frames_) {
    fprintf(stderr, "Track::SetTimes: input `times` has different number "
                "of frames (%d != %d)", length, num_frames_);
  }
  for (int i = 0; i < num_frames_; ++i) {
    data_.Set(0, i, times[i]);
  }
}

bool Track::SetVoicing(const std::vector<bool> &vuv) {
  if (vuv.size() != num_frames_) {
    fprintf(stderr, "Track::SetVoicing: input has different number "
            "of frames (%zu != %d)", vuv.size(), num_frames_);
    return false;
  }
  val_ = vuv;
  return true;
}

void Track::FrameOut(std::vector<float> *fv, int n) const {
  fv->resize(num_channels_);
  for (int i = 0; i < num_channels_; i++) {
    (*fv)[i] = a(n, i);
  }
}

void Track::FrameOut(std::vector<double> *fv, int n) const {
  fv->resize(num_channels_);
  for (int i = 0; i < num_channels_; i++) {
    (*fv)[i] = a(n, i);
  }
}

void Track::FrameIn(const float &f, int n) {
  a(n) = f;
}

void Track::FrameIn(const std::vector<float> &fv, int n) {
  for (int i = 0; i < num_channels_; i++) {
    a(n, i) = fv[i];
  }
}

void Track::FrameIn(const std::vector<double> &fv, int n) {
  for (int i = 0; i < num_channels_; i++) {
    a(n, i) = fv[i];
  }
}

Track *Track::GetSubTrack(float start,
                          float end,
                          int ch_offset,
                          int ch_size) const {
  int start_frame = -1;
  int end_frame = -1;
  int n_sub_frames;
  int i, j;

  for (i = 0; i < this->num_frames(); ++i) {
    if ((start_frame == -1) && (this->t(i) > start)) {
      start_frame = i - 1;
    }
    if ((end_frame == -1) && (this->t(i) > end)) {
      end_frame = i;
    }
  }
  if (start_frame == -1) {
    start_frame = this->num_frames() - 1;
  }
  if (end_frame == -1) {
    end_frame = this->num_frames() - 1;
  }
  n_sub_frames = end_frame - start_frame;
  if (n_sub_frames < 1) {
    // make sure we have at least one frame
    n_sub_frames = 1;
  }
  if (ch_size == -1 && ch_offset == -1) {
    ch_offset = 0;
    ch_size = this->num_channels();
  } else {
    if (ch_size > this->num_channels()) {
      fprintf(stderr, "Incorrect number of channels for sub track");
      return NULL;
    }
  }
  Track *sub_track = new Track;
  sub_track->resize(n_sub_frames, ch_size);
  for (i = 0; i < n_sub_frames; ++i) {
    sub_track->t(i) = this->t(i + start_frame);
    for (j = ch_offset; j < ch_size; ++j) {
      sub_track->a(i, j) = this->a(i + start_frame, j);
    }
  }
  return sub_track;
}


Track *Track::GetSubTrack(int start_frame_index,
                          int end_frame_index,
                          int start_channel_index,
                          int end_channel_index) const {
  if (start_frame_index > end_frame_index ||
      start_frame_index < 0 || start_frame_index >= this->num_frames() ||
      end_frame_index < 0   || end_frame_index >= this->num_frames()) {
    fprintf(stderr, "Incorrect frame indices for sub-track 0<%d<=%d<%d\n",
                start_frame_index,
                end_frame_index,
                this->num_frames());
    return NULL;
  }

  if (start_channel_index > end_channel_index ||
      start_channel_index < 0 || start_channel_index >= this->num_channels() ||
      end_channel_index < 0   || end_channel_index >= this->num_channels()) {
    fprintf(stderr, "Incorrect channel indices for sub-track 0<%d<=%d<%d\n",
                start_channel_index,
                end_channel_index,
                this->num_channels());
    return NULL;
  }

  Track *sub_track = new Track;

  int number_frames = end_frame_index - start_frame_index + 1;
  int number_channels = end_channel_index - start_channel_index + 1;
  sub_track->resize(number_frames, number_channels);

  for (int f = 0; f < number_frames; f++) {
    for (int p = 0; p < number_channels; p++) {
      sub_track->a(f, p) = this->a(start_frame_index + f,
                                   start_channel_index + p);
    }
    sub_track->t(f) = this->t(start_frame_index + f);
    sub_track->set_v(f, this->v(start_frame_index + f));
  }

  sub_track->set_shift(this->shift());
  return sub_track;
}

void Track::resize(int n, int c) {
  if (n != num_frames_) {
    val_.resize(n, 1);
  }

  data_.resize(c + 1, n);
  num_frames_ = n;
  num_channels_ = c;
}

void Track::Clear() {
  num_frames_ = 0;
  num_channels_ = 0;
  shift_ = 0.0F;
  data_.clear();
  val_.clear();
}

void Track::CopyFrom(const char *v, int32_t num_frames, int32_t num_channels) {
  resize(num_frames, num_channels);
  const float* source = reinterpret_cast<const float*>(v);
  for (int32_t i = 0; i < num_frames; ++i) {
    for (int32_t j = 0; j < num_channels; ++j) {
      a(i, j) = *source++;
    }
  }
}

// Sets current track to be combination of two tracks
bool Track::SetCombinedTrack(const Track &track_a, const Track &track_b) {
  if (track_a.num_frames() != track_b.num_frames()) {
    fprintf(stderr, "Mismatching number of frames: %d vs. %d",
                track_a.num_frames(),
                track_b.num_frames());
    return false;
  }

  const int n_frames = track_a.num_frames();
  const int n_channels_a = track_a.num_channels();
  const int n_channels_b = track_b.num_channels();
  resize(n_frames, n_channels_a + n_channels_b);
  for (int i = 0; i < n_frames; i++) {
    for (int j = 0; j < n_channels_a; j++) {
      a(i, j) = track_a.a(i, j);
    }
    for (int j = 0; j < n_channels_b; j++) {
      a(i, n_channels_a + j) = track_b.a(i, j);
    }
    set_v(i, track_a.v(i) && track_b.v(i));
  }
  return true;
}

bool Track::SetCombinedTrack(const Track &track_a,
                             const Track &track_b,
                             const Track &track_c) {
  Track track_ab;
  if (!track_ab.SetCombinedTrack(track_a, track_b)) {
    return false;
  }
  return SetCombinedTrack(track_ab, track_c);
}

// Pads a track with <num_pads> frames:
bool Track::Pad(int num_pads) {
  if (num_pads <= 0) {
    return false;
  }
  const int num_original_frames = num_frames();
  const int num_original_channels = num_channels();

  Track new_track;
  new_track.resize(num_original_frames + num_pads, num_original_channels);
  for (int i = 0; i < num_original_frames; ++i) {
    for (int j = 0; j < num_original_channels; ++j) {
      new_track.a(i, j) = a(i, j);
    }
    new_track.set_v(i, v(i));
  }
  for (int i = 0; i < num_pads; ++i) {
    for (int j = 0; j < num_original_channels; ++j) {
      new_track.a(num_original_frames + i, j) = a(num_original_frames - 1, j);
    }
    new_track.set_v(i, v(num_original_frames - 1));
  }

  const int num_new_frames = new_track.num_frames();
  Clear();
  resize(num_new_frames, num_original_channels);
  for (int i = 0; i < num_new_frames; ++i) {
    for (int j = 0; j < num_original_channels; ++j) {
      a(i, j) = new_track.a(i, j);
    }
    set_v(i, new_track.v(i));
  }
  return true;
}

// Forces the two tracks to have the same size, padding if necessary.
void Track::MakeSameSize(Track *track) {
  int max_frames = num_frames();
  if (track->num_frames() > max_frames) {
    max_frames = track->num_frames();
  }
  Pad(max_frames - num_frames());
  track->Pad(max_frames - track->num_frames());
}

// Forces the three tracks to have same size, padding if necessary.
void Track::MakeSameSize(Track *track_a, Track *track_b) {
  int max_frames = num_frames();
  if (track_a->num_frames() > max_frames) {
    max_frames = track_a->num_frames();
  }
  if (track_b->num_frames() > max_frames) {
    max_frames = track_b->num_frames();
  }
  Pad(max_frames - num_frames());
  track_a->Pad(max_frames - track_a->num_frames());
  track_b->Pad(max_frames - track_b->num_frames());
}

//------------------------------------------------------------------------------
// Utils.
//------------------------------------------------------------------------------

void ConvertToLogarithmic(Track *t) {
  for (int32_t i = 0; i < t->num_frames(); ++i) {
    for (int32_t j = 0; j < t->num_channels(); ++j) {
      if (t->a(i, j) <= 0.0F) {
        t->a(i, j) = INVALID_LOG;
      } else {
        t->a(i, j) = log(t->a(i, j));
      }
    }
  }
}

float *TrackToFloatPointer(const Track &track, int *num_samples) {
  *num_samples = track.num_channels() * track.num_frames();
  float *array = new float[*num_samples];
  uint32_t k = 0;
  for (int i = 0; i < track.num_frames(); ++i) {
    for (int j = 0; j < track.num_channels(); ++j) {
      array[k++] = track.a(i, j);
    }
  }
  return array;
}
