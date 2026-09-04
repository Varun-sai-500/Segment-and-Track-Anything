Third-Party Notices

This project incorporates or depends on third-party models and source code. The following notices are provided in accordance with the licensing terms of those components.

1. Segment Anything Model (SAM)
Component: facebook/sam-vit-base
Source: https://huggingface.co/facebook/sam-vit-base
Provider: Meta AI (FAIR)
Usage in this project: Loaded via the Hugging Face transformers library for segmentation.
License: Apache License 2.0
Copyright: Copyright (c) Meta Platforms, Inc. and affiliates.

> Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

> http://www.apache.org/licenses/LICENSE-2.0

> Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

2. Grounding DINO
Component: IDEA-Research/grounding-dino-base
Model repository: https://huggingface.co/IDEA-Research/grounding-dino-base
Provider: IDEA Research
Usage in this project: Loaded via the Hugging Face transformers library for open-vocabulary object detection.
License: Apache License 2.0
Copyright: Copyright (c) IDEA Research.

> Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

> http://www.apache.org/licenses/LICENSE-2.0

> Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

3. DeAOT Tracker (adapted from AOT-Benchmark)
Component: DeAOT tracking implementation, adapted from source
Original repository: https://github.com/yoxu515/aot-benchmark
Copyright: Copyright (c) 2020, z-x-yang
License: BSD 3-Clause License
Modifications: The implementation in this repository has been substantially modified from the upstream source. It has been trimmed and specialized to support a single model configuration rather than the general-purpose, multi-model design of the original codebase. As such, this is a derivative work of the original AOT-Benchmark project.

> BSD 3-Clause License

> Copyright (c) 2020, z-x-yang All rights reserved.

> Redistribution and use in source and binary forms, with or without modification, are permitted  provided that the following conditions are met:

> Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
> Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
> Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE