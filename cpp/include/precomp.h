#pragma once

#include <stdio.h>
#include <stdlib.h>


#include <string>
#include <vector>
#include <iostream>
#include <sstream>
using namespace std;


#include "yaml-cpp/yaml.h"
#include "feat/fbank.h"
#include "feat/feature_pipeline.h"
#include "feat/wav.h"
#include "feat/fft.h"

#include "ctc_beam_search_decoder.h"

#include "onnxruntime/onnxruntime_run_options_config_keys.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

// user defined headers
#include "rpdatadef.h"

#include "commfunc.h"
#include "librpasrapi.h"
#include "rapidasr.h"


