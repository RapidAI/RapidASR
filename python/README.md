<div align="center">
<div align="center">
    <h1><b>Rapid Paraformer</b></h1>
</div>

<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.13-aff.svg"></a>
<a href="https://pepy.tech/project/rapid_paraformer"><img src="https://static.pepy.tech/personalized-badge/rapid_paraformer?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
<a href="https://pypi.org/project/rapid_paraformer/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid_paraformer"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</div>

## 简介
rapid_paraformer是一个基于阿里达摩院[Paraformer语音识别-中文-通用-16k-离线-large-pytorch](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)的语音识别工具。

🎉该项目核心代码已经并入[FunASR](https://github.com/alibaba-damo-academy/FunASR)

本仓库仅对模型做了转换，只采用ONNXRuntime推理引擎

## TODO
- [ ] 整合vad + asr + pun三个模型，打造可部署使用的方案

## 安装
```bash
pip install rapid_paraformer
```

## 模型下载
方法一：从Hugging Face上下载（[link](https://huggingface.co/SWHL/RapidParaformer)）
```python
from rapid_paraformer import download_hf_model

download_hf_model(repo_id="SWHL/RapidParaformer", save_dir=".")
```

方法二：([Google Drive](https://drive.google.com/drive/folders/1RVQtMe0eB_k6G5TJlmXwPELx4VtF2oCw?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1zf8Ta6QxFHY3Z75fHNYKrQ?pwd=6ekq))
```bash
resources
├── [ 700]  config.yaml
└── [4.0K]  models
    ├── [ 11K]  am.mvn
    ├── [824M]  asr_paraformerv2.onnx
    └── [ 50K]  token_list.pkl
```

## 模型转换
基于modescope下的notebook环境自助转换：
1. 打开[快速体验](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
2. 打开notebook → Cell中输入以下命令, 执行即可。
    ```bash
    !python -m funasr.export.export_model --model-name 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' --export-dir "./export"
    ```

## 使用
```python
from rapid_paraformer import RapidParaformer

config_path = "resources/config.yaml"

paraformer = RapidParaformer(config_path)

wav_path = [
    "test_wavs/0478_00017.wav",
    "test_wavs/asr_example_zh.wav",
]

result = paraformer(wav_path)
print(result)
# ['y', '欢迎大家来体验达摩院推出的语音识别模型']
```
