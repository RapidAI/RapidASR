## Rapid ASR

<p align="left">
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

- ASR模型出自阿里达摩院[Paraformer语音识别-中文-通用-16k-离线-large-pytorch](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
- VAD模型FSMN-VAD出自阿里达摩院[FSMN语音端点检测-中文-通用-16k](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
- Punc模型CT-Transformer出自阿里达摩院[CT-Transformer标点-中文-通用-pytorch](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)
- 🎉该项目核心代码已经并入[FunASR](https://github.com/alibaba-damo-academy/FunASR)
- 本仓库仅对模型做了转换，只采用ONNXRuntime推理引擎

#### TODO
- [ ] 整合vad + asr + pun三个模型，打造可部署使用的方案

#### 使用步骤
1. 安装环境
   ```bash
    pip install -r requirements.txt
   ```
2. 下载模型
   - 由于模型太大（823.8M），上传到仓库不容易下载，
        - （推荐）自助转换：基于modescope下的notebook环境，可一键转换，详情戳：[快速体验](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
            - 打开notebook → Cell中输入`!python -m funasr.export.export_model 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" true`, 执行即可。
        - 提供百度云下载连接：[asr_paraformerv2.onnx](https://pan.baidu.com/s/1-nEf2eUpkzlcRqiYEwub2A?pwd=dcr3)（模型MD5: `9ca331381a470bc4458cc6c0b0b165de`）
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
    ```


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

更新内容：

1、更新了VAD和Punc 

更新内容主要代码都来源于[FunASR](https://github.com/alibaba-damo-academy/FunASR) 

模型导出参考[这里](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export) ，把导出来的model.onnx放到对应的文件夹就可以了。

demo里面组合了使用方式 ，目前来看VAD的效果不太好，所以我这里直接是把音频手动按固定的30s切了，然后再去识别组合。

