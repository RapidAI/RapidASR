## Rapid ASR
### 商用级开源语音自动识别程序库，开箱即用，全平台支持，中英文混合识别。

<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/C++-aff.svg"></a>
</p>

- 模型出自阿里达摩院[Paraformer语音识别-中文-通用-16k-离线-large-pytorch](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
- 🎉该项目核心代码已经并入[FunASR](https://github.com/alibaba-damo-academy/FunASR)
- 本仓库仅对模型做了转换，只采用ONNXRuntime推理引擎

#### 文档导航
- [Python版](./python/README.md)
- [C++/C版](./cpp_onnx/readme.md)

#### TODO
- [ ] 整合vad + asr + pun三个模型，打造可部署使用的方案


#### 更新日志
- 2023-02-25
   - 添加C++版本推理，使用onnxruntime引擎，预/后处理代码来自： https://github.com/chenkui164/FastASR

- 2023-02-14 v2.0.3 update:
  - 修复librosa读取wav文件错误
  - 修复fbank与torch下fbank提取结果不一致bug

- 2023-02-11 v2.0.2 update:
  - 模型和推理代码解耦（`rapid_paraformer`和`resources`）
  - 支持批量推理（通过`resources/config.yaml`中`batch_size`指定）
  - 增加多种输入方式（`Union[str, np.ndarray, List[str]]`）

- 2023-02-10 v2.0.1 update:
  - 添加对输入音频为噪音或者静音的文件推理结果捕捉。
