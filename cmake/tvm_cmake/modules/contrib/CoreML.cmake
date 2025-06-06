# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_COREML)
  message(STATUS "Build with contrib.coreml")
  find_library(FOUNDATION_LIB Foundation)
  find_library(COREML_LIB Coreml)
  file(GLOB COREML_CONTRIB_SRC 3rdparty/tvm/src/runtime/contrib/coreml/*.mm)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${FOUNDATION_LIB} ${COREML_LIB})
  list(APPEND RUNTIME_SRCS ${COREML_CONTRIB_SRC})
endif(USE_COREML)
