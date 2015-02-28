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

#include "core/file_resource.h"

FileResource::FileResource(const std::string &filename, const std::string &mode)
    : filename_(filename), mode_(mode) {
}

FileResource::~FileResource() {
  fclose(fp_);
}

bool FileResource::Get() {
  fp_ = fopen(filename_.c_str(), mode_.c_str());
  return fp_ != NULL;
}
