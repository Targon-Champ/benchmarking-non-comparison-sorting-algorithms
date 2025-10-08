@echo off
echo Compiling dataset generation framework...

REM This assumes you have Visual Studio tools or g++ installed and in your PATH
REM If you have Visual Studio, uncomment the following line and comment the g++ line:
REM cl /EHsc /I src/common src/common/dataset_main.cpp src/common/dataset_generator.cpp src/common/dataset_validator.cpp /Fe:dataset_main.exe

REM If you have g++, use this line:
g++ -std=c++17 -I src/common src/common/dataset_main.cpp src/common/dataset_generator.cpp src/common/dataset_validator.cpp -o dataset_main.exe

if %ERRORLEVEL% EQU 0 (
    echo Compilation successful!
    echo Running dataset generation...
    mkdir datasets 2>nul
    mkdir results 2>nul
    dataset_main.exe
) else (
    echo Compilation failed. Please ensure you have a C++ compiler installed (like Visual Studio, MinGW, or Cygwin).
    echo Make sure that the compiler is in your system PATH.
)