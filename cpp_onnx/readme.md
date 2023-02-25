


## 快速使用

Windows 下下载onnxruntime库，解开到某个位置，指定onnxruntime目录按以下方式：
```

cmake  --config release -DONNXRUNTIME_DIR=D:\\thirdpart\\onnxruntime

````
使用  -DONNXRUNTIME_DIR  指向onnxruntime目录，该目录下有include以及lib
```

├───include
└───lib
```
Windows下已经预置fftw3及openblas库。

## 支持平台

- Windows
- Linux/Unix

## 依赖
- fftw3
- openblas
- onnxruntime


## 导出onnx格式模型文件
安装 modelscope与FunASR，[安装文档](https://github.com/alibaba-damo-academy/FunASR/wiki)
```shell
pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple #推荐使用上交pip源
pip install "modelscope[audio_asr]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install --editable ./
```
导出onnx模型，[详见](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)，参考示例，从modelscope中模型导出：

```
python -m funasr.export.export_model 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" true
```
