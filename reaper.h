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
/*
Author: Kai Zhao (loverszhao@gmail.com)
*/

#ifndef _REAPER_H_
#define _REAPER_H_

// A macro to disallow operator=
// This should be used in the private: declarations for a class.
#define DISALLOW_ASSIGN(TypeName)\
  void operator=(TypeName const &)

// A macro to disallow copy constructor and operator=
// This should be used in the private: declarations for a class.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  DISALLOW_ASSIGN(TypeName)

#endif  // _REAPER_H_
