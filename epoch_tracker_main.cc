/*
Copyright 2014 Google Inc. All rights reserved.

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

#include <memory>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "core/file_resource.h"
#include "core/track.h"
#include "epoch_tracker/epoch_tracker.h"
#include "wave/wave.h"


const char* kHelp = "Usage: <bin> -i <input_file> "
    "[-f <f0_output> -p <pitchmarks_output> "
    "-t "
    "-s "
    "-e <float> "
    "-x <float> "
    "-m <float> "
    "-u <float> "
    "-a "
    "-d <debug_output_basename>] "
    "\n\n Help:\n"
    "-t enables a Hilbert transform that may reduce phase distortion\n"
    "-s suppress applying high pass filter at 80Hz "
    "(rumble-removal highpass filter)\n"
    "-e specifies the output frame interval for F0\n"
    "-x maximum f0 to look for\n"
    "-m minimum f0 to look for\n"
    "-u regular inter-pulse interval to use in unvoiced regions\n"
    "-a saves F0 and PM output in ascii mode\n"
    "-d write diagnostic output to this file pattern\n";

int main(int argc, char* argv[]) {
  int opt = 0;
  std::string filename;
  std::string f0_output;
  std::string pm_output;
  bool do_hilbert_transform = kDoHilbertTransform;
  bool do_high_pass = kDoHighpass;
  float external_frame_interval = kExternalFrameInterval;
  float max_f0 = kMaxF0Search;
  float min_f0 = kMinF0Search;
  float inter_pulse = kUnvoicedPulseInterval;
  bool ascii = false;
  std::string debug_output;
  while ((opt = getopt(argc, argv, "i:f:p:htse:x:m:u:ad:")) != -1) {
    switch(opt) {
      case 'i':
        filename = optarg;
        break;
      case 'f':
        f0_output = optarg;
        break;
      case 'p':
        pm_output = optarg;
        break;
      case 't':
        do_hilbert_transform = true;
        break;
      case 's':
        do_high_pass = false;
        break;
      case 'e':
        external_frame_interval = atof(optarg);
        break;
      case 'x':
        max_f0 = atof(optarg);
        break;
      case 'm':
        min_f0 = atof(optarg);
        break;
      case 'u':
        inter_pulse = atof(optarg);
        break;
      case 'a':
        ascii = true;
        break;
      case 'd':
        debug_output = optarg;
        break;
      case 'h':
        fprintf(stdout, "\n%s\n", kHelp);
        return 0;
    }
  }

  // Load input.
  Wave wav;
  if (!wav.Load(filename)) {
    fprintf(stderr, "Failed to load waveform '%s'\n", filename.c_str());
    return 1;
  }

  // Compute f0 and pitchmarks.
  EpochTracker et;
  if (!et.Init()) {
    return 1;
  }
  if (!debug_output.empty()) {
    et.set_debug_name(debug_output);
  }
  et.set_do_hilbert_transform(do_hilbert_transform);
  et.set_do_highpass(do_high_pass);
  et.set_external_frame_interval(external_frame_interval);
  et.set_max_f0_search(max_f0);
  et.set_min_f0_search(min_f0);
  et.set_unvoiced_pulse_interval(inter_pulse);

  Track *f0 = NULL;
  Track *pm = NULL;
  if (!et.ComputeEpochs(wav, &pm, &f0)) {
    fprintf(stderr, "Failed to compute epochs\n");
    return 1;
  }

  // Save outputs.
  if (!f0_output.empty() && !f0->Save(f0_output, ascii)) {
    delete f0;
    fprintf(stderr, "Failed to save f0 to '%s'\n", f0_output.c_str());
    return 1;
  }
  if (!pm_output.empty() && !pm->Save(pm_output, ascii)) {
    delete pm;
    fprintf(stderr, "Failed to save pitchmarks to '%s'\n", pm_output.c_str());
    return 1;
  }
  delete f0;
  delete pm;
  return 0;
}
