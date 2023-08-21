## rapid_paraformer

<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
    <a href="https://pepy.tech/project/rapid_paraformer"><img src="https://static.pepy.tech/personalized-badge/rapid_paraformer?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
    <a href="https://pypi.org/project/rapid_paraformer/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid_paraformer"></a>
    <a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


### Use
1. Install
    1. Install the `rapid_paraformer`
        ```bash
        pip install rapid_paraformer
        ```
    2. Download the **resources.zip** ([Google Drive](https://drive.google.com/drive/folders/1RVQtMe0eB_k6G5TJlmXwPELx4VtF2oCw?usp=sharing) | [Baidu NetDisk](https://pan.baidu.com/s/1zf8Ta6QxFHY3Z75fHNYKrQ?pwd=6ekq))
        ```bash
        resources
        ├── [ 700]  config.yaml
        └── [4.0K]  models
            ├── [ 11K]  am.mvn
            ├── [824M]  asr_paraformerv2.onnx
            └── [ 50K]  token_list.pkl
        ```
2. Use
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
    ```

### See details for [RapidASR](https://github.com/RapidAI/RapidASR).
