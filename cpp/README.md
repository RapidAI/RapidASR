## RapidASR CPP
- Our vision is to offer an out-of-box engineering implementation for ASR
- A cpp implementation of recognize-onnx.py in [Wenet-asr](https://github.com/wenet-e2e/wenet) in which it implements the inference with ONNXRuntime.
- For a version of pure CPP code, we need to do a bit of work to rewrite some components.
- Special thanks to its original author SlyneD.
- Less is more. Less dependency, more usability.
- Just offline mode, not support stream mode, aka separate files can be recognized.
- **QQ Group: 645751008**

### Supported modes:
- CTC_GREEDY_SEARCH
- CTC_RPEFIX_BEAM_SEARCH
- ATTENSION_RESCORING

### Models
- The model is original from [wenetspeech/s0](https://github.com/wenet-e2e/wenet/tree/main/examples/wenetspeech/s0) and tested with `recognize-onnx.py`.
- Download [Bidirectional model](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/wenetspeech/20211025_conformer_bidecoder_exp.tar.gz)
- Download:
    - URL：https://pan.baidu.com/s/1BTR-uR_8WWBFpvOisNR_PA
    - Extract code：9xjz
- Sample Rate: 16000Hz
- Sample Depth: 16bits
- Channel: single

### Build
- Windows
    ```
    Visual studio 2019 & cmake 3.20

    cd thirdpart
    build_win.cmd x86|x64
    ```
- Linux
    ```
    cmake
    ```