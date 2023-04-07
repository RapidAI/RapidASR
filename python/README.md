#### 基于Wenet训练所得模型的推理代码
- 项目来源：[wenet/aishell/s0](https://github.com/wenet-e2e/wenet/blob/main/examples/aishell/s0/README.md)
- 运行环境：Python 3.7 | CPU | 不依赖torch和torchaudio

#### 使用方法
1. 下载整个`python/base_wenet`目录

2. 安装依赖环境
   - 安装依赖包
        ```bash
        pip install -r requirements.txt -i https://pypi.douban.com/simple/
        ```
    - 编译安装[`ctc_decoders`](https://github.com/Slyne/ctc_decoder)
        ```bash
        git clone https://github.com/Slyne/ctc_decoder.git
        apt-get update
        apt-get install swig
        apt-get install python3-dev
        cd ctc_decoder/swig && bash setup.sh
        ```

3. 下载预训练onnx模型到`pretrain_model\20211025_conformer_exp`下,
    - 下载链接：[Google Drive](https://drive.google.com/drive/folders/1Jv9pi44McsGfpFrK9R8zm9ZJuVzlP-uL?usp=sharing)
    - 最终结构目录如下，请自行比对：
        ```text
        .
        ├── pretrain_model
        │   └── 20211025_conformer_exp
        │       ├── decoder.onnx
        │       ├── encoder.onnx
        │       ├── test.yaml
        │       └── words.txt
        ├── README.md
        ├── requirements.txt
        ├── test_data
        │   └── test.wav
        ├── test_demo.py
        └── wenet
            ├── __init__.py
            ├── kaldifeat
            │   ├── feature.py
            │   ├── __init__.py
            │   ├── ivector.py
            │   ├── LICENSE
            │   └── README.md
            ├── utils.py
            └── wenet_infer.py
        ```


4. 运行`python test_demo.py`
5. 运行结果如下：
   ```text
    test_data/test.wav      甚至出现交易几乎停滞的情况      0.8272988796234131s
   ```

#### 模型转onnx代码
- 原始的Wenet模型下载路径：[20211025_conformer_exp](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20211025_conformer_exp.tar.gz)
- 训练该模型使用的配置为：[train_conformer.yaml](https://github.com/wenet-e2e/wenet/blob/a92952827c/examples/aishell/s0/conf/train_conformer.yaml)
```bash
# 运行环境：python3.7 torch1.10 1个GPU 16CPU 内存32G
root_dir="examples/mix_data/exp/conformer/2022-04-07-05-37-18"
config_path="${root_dir}/train.yaml"
cmvn_file="${root_dir}/global_cmvn"
checkpoint_path="${root_dir}/4.pt"
dir_name=${checkpoint_path##*/}
dir_name=${dir_name%.*}
out_onnx_dir="export_onnx/mix_data/${dir_name}"

python wenet/bin/export_onnx.py --config ${config_path} \
                      --checkpoint ${checkpoint_path} \
                      --cmvn_file ${cmvn_file} \
                      --output_onnx_dir ${out_onnx_dir}

```
