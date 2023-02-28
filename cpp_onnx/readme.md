

## 注意
本程序只支持 采样率16000hz, 位深16bit的 **单声道** 音频。

## 快速使用

### Windows
 
 安装Vs2022 打开cpp_onnx目录下的cmake工程，直接 build即可。 本仓库已经准备好所有相关依赖库。
 
 Windows下已经预置fftw3、onnxruntime及openblas库


### Linux
See the bottom of this page: Building Guidance





###  运行程序

tester  /path/to/models/dir /path/to/wave/file

 例如： tester /data/models  /data/test.wav

/data/models 需要包括如下两个文件： model.onnx 和vocab.txt

```

```
## 支持平台

- Windows
- Linux/Unix

## 依赖
- fftw3
- openblas
- onnxruntime


## 导出onnx格式模型文件
安装 modelscope与FunASR，依赖：torch，torchaudio，安装[详细参考文档]([安装文档](https://github.com/alibaba-damo-academy/FunASR/wiki))
```shell
pip install "modelscope[audio_asr]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install --editable ./
```
导出onnx模型，[详见](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)，参考示例，从modelscope中模型导出：

```
python -m funasr.export.export_model 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch' "./export" true
```

## Building Guidance for Linux/Unix

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

### The structure of a qualified onnxruntime package.
```
onnxruntime_xxx
├───include
└───lib
```

