## Rapid paraformer

<p align="left">
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

- 模型出自阿里达摩院[Paraformer语音识别-中文-通用-16k-离线-large-pytorch](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
- 🎉该项目核心代码已经并入[FunASR](https://github.com/alibaba-damo-academy/FunASR)
- 本仓库仅对模型做了转换，只采用ONNXRuntime推理引擎


#### 更新日志
- 2024-02-25 
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


#### 使用步骤
1. 安装环境
   ```bash
    pip install -r requirements.txt
   ```
2. 下载模型
   - 由于模型太大（823.8M），上传到仓库不容易下载，提供百度云下载连接：[asr_paraformerv2.onnx](https://pan.baidu.com/s/1-nEf2eUpkzlcRqiYEwub2A?pwd=dcr3)（模型MD5: `9ca331381a470bc4458cc6c0b0b165de`）
   - 模型下载之后，放在`resources/models`目录下即可，最终目录结构如下：
        ```text
        .
        ├── demo.py
        ├── rapid_paraformer
        │   ├── __init__.py
        │   ├── kaldifeat
        │   ├── __pycache__
        │   ├── rapid_paraformer.py
        │   └── utils.py
        ├── README.md
        ├── requirements.txt
        ├── resources
        │   ├── config.yaml
        │   └── models
        │       ├── am.mvn
        │       ├── asr_paraformerv2.onnx  # 放在这里
        │       └── token_list.pkl
        ├── test_onnx.py
        ├── tests
        │   ├── __pycache__
        │   └── test_infer.py
        └── test_wavs
            ├── 0478_00017.wav
            └── asr_example_zh.wav
        ```

3. 运行demo
    ```python
    from rapid_paraformer import RapidParaformer


    config_path = 'resources/config.yaml'
    paraformer = RapidParaformer(config_path)

    # 输入：支持Union[str, np.ndarray, List[str]] 三种方式传入
    # 输出： List[asr_res]
    wav_path = [
        'test_wavs/0478_00017.wav',
    ]

    result = paraformer(wav_path)
    print(result)
    ```
4. 查看结果
   ```text
   ['呃说不配合就不配合的好以上的话呢我们摘取八九十三条因为这三条的话呢比较典型啊一些数字比较明确尤其是时间那么我们要投资者就是了解这一点啊不要轻信这个市场可以快速回来啊这些配市公司啊后期又利好了可
   以快速快速攻能包括像前一段时间啊有些媒体在二三月份的时候']
   ```
