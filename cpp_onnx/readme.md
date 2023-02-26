


## 快速使用

Windows 下下载onnxruntime库，解开到某个位置，指定onnxruntime目录按以下方式：
```

cmake  --config release -DONNXRUNTIME_DIR=D:\\thirdpart\\onnxruntime

````
使用  -DONNXRUNTIME_DIR  指向onnxruntime目录，该目录下有include以及lib
```
onnxruntime_xxx
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

## Building Guidance

```
git clone https://github.com/RapidAI/RapidASR.git
cd RapidASR/cpp_onnx/
mkdir build
cd build
# download an appropriate onnxruntime from https://github.com/microsoft/onnxruntime/releases/tag/v1.14.0
# here we get a copy of onnxruntime for linux 64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-1.14.0.tgz
#ls 
# onnxruntime-linux-x64-1.14.0  onnxruntime-linux-x64-1.14.0.tgz

#install fftw3-dev
apt install libfftw3-dev
#install openblas
apt install libopenblas-dev

# build 
 cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/mnt/c/Users/ma139/RapidASR/cpp_onnx/build/onnxruntime-linux-x64-1.14.0
 make
 
 # then in the subfolder tester of current direcotry, you will see a program, tester
 

````