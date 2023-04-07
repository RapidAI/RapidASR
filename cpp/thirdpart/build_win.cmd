@echo off
set CURDIR=%cd%

rem win32 |x64

call:CompileLib %CURDIR% x64 

cd %CURDIR%
call:CompileLib %CURDIR% win32

cd %CURDIR%


rem  编译函数

:CompileLib
cd %~1
cd openfst
if exist build  ( rd  /q /s build )
mkdir build
cd build
cmake ..  -A  %~2 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../winlib/%~2
cmake --build .  --config Release -j8
GOTO:EOF


