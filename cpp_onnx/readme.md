
## 快速使用

Windows 下下载onnxruntime库，解开到某个位置，指定onnxruntime目录按以下方式：

cmake  --config release -DONNXRUNTIME_DIR=D:\\thirdpart\\onnxruntime

使用  -DONNXRUNTIME_DIR  指向onnxruntime目录，该目录下有include以及lib

Windows下已经预置fftw3及openblas库。

`
├───include

 └───lib
`
## 支持平台

- Windows
- Linux/Unix

## 依赖
- fftw3
- openblas
