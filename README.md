## Rapid paraformer

<p align="left">
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

- 模型出自阿里达摩院[Paraformer语音识别-中文-通用-16k-离线-large-pytorch](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
- 本仓库仅对模型做了转换，只采用ONNXRuntime推理引擎


#### 更新日志
- 2023-02-10 v2.0.1 update:
  - 添加对输入音频为噪音或者静音的文件推理结果捕捉。


#### 使用步骤
1. 安装环境
   ```bash
    pip install -r requirements.txt
   ```
2. 下载模型
   - 由于模型太大（881M），上传到仓库不容易下载，提供百度云下载连接：[asr_paraformerv2.onnx](https://pan.baidu.com/s/1-nEf2eUpkzlcRqiYEwub2A?pwd=dcr3)
   - 模型下载之后，放在`rapid_paraformer/models`目录下即可，最终目录结构如下：
        ```text
        rapid_paraformer
        ├── config.yaml
        ├── __init__.py
        ├── kaldifeat
        │   ├── feature.py
        │   ├── __init__.py
        │   ├── ivector.py
        │   ├── LICENSE
        │   └── README.md
        ├── models
        │   ├── am.mvn
        │   ├── asr_paraformerv2.onnx  # 放在这里
        │   └── token_list.pkl
        ├── rapid_paraformer.py
        └── utils.py
        ```

3. 运行demo
    ```python
    from rapid_paraformer import RapidParaformer

    paraformer = RapidParaformer()

    wav_path = 'test_wavs/example_test.wav'
    result = paraformer(str(wav_path))
    print(result)
    ```
4. 查看结果
   ```text
   [['呃说不配合就不配合的好以上的话呢我们摘取八九十三条因为这三条的话呢比较典型啊一些数字比较明确尤其是时间那么我们要投资者就是了解这一点啊不要轻信这个市场可以快速回来啊这些配市公司啊后期又利好了可
   以快速快速攻能包括像前一段时间啊有些媒体在二三月份的时候']]
   ```
