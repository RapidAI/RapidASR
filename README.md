#### 基于PaddeSpeech训练所得模型的推理代码
- 项目来源：[PaddleSpeech/s2t](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell/asr0)
- 运行环境：Linux| Python 3.7 | CPU | 不依赖Paddle

#### 使用方法
1. 下载整个`python/base_paddlespeech`目录

2. 安装依赖环境
   - 批量安装
    ```bash
    pip install -r requirements.txt -i https://pypi.douban.com/simple/
    
    # CentOS
    sudo yum install libsndfile 
    ```
3. 下载`resources`模型相关文件到`base_paddlespeech`下,
    - 下载`resources`链接：[Google Drive](https://drive.google.com/file/d/1MWmKxsfCNQyQ5CPlaYxJKnYfIIC5OO5L/view?usp=sharing)
    - 下载语言模型文件→[下载链接](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm)，放到`base_paddlespeech/resources/models/language_model`目录下
    - 最终结构目录如下，请自行比对：
        ```text
        base_paddlespeech
        ├── deepspeech2
        │   ├── infer.py
        │   ├── __init__.py
        │   └── s2t
        │       ├── decoders
        │       ├── deepspeech2.py
        │       ├── frontend
        │       ├── io
        │       ├── modules
        │       ├── __pycache__
        │       ├── transform
        │       └── utils
        ├── main.py
        ├── requirements.txt
        ├── resources
        │   └── models
        │       ├── asr0_deepspeech2_online_aishell_ckpt_0.2.0.onnx
        │       ├── language_model
        │       │   └── zh_giga.no_cna_cmn.prune01244.klm
        │       └── model.yaml
        └── test_wav
            └── zh.wav
        ```


4. 运行`python main.py`
5. 运行结果如下：
   ```text
    checking the audio file format......
    The sample rate is 16000
    The audio file format is right
    Preprocess audio_file:/da2/SWHL/test_wav/zh.wav
    audio feat shape: (1, 498, 161)
    ASR Result:     我认为跑步最重要的就是给我们带来了身体健康
   ```

#### 模型转onnx代码
```bash
model_dir="pretrained_models/deepspeech2online_aishell-zh-16k/asr0_deepspeech2_online_aishell_ckpt_0.1.1.model.tar/exp/deepspeech2_online/checkpoints"
pdmodel="avg_1.jit.pdmodel"
params_file="avg_1.jit.pdiparams"
save_onnx="pretrained_models/onnx/asr0_deepspeech2_online_aishell_ckpt_0.1.1.onnx"

paddle2onnx --model_dir ${model_dir} \
            --model_filename ${pdmodel} \
            --params_filename ${params_file} \
            --save_file ${save_onnx} \
            --opset_version 12
```
