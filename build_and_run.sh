#!/bin/bash

set -e

echo "===================================================="
echo " Starting Non-Invasive AI Telemetry Platform Build "
echo "===================================================="
if [ -d "build" ]; then
    echo "[*] Cleaning existing build directory..."
    rm -rf build
fi

echo "[*] Configuring CMake build system (Release mode)..."
cmake -B build -DCMAKE_BUILD_TYPE=Release

echo "[*] Compiling ai_telemetry_tool..."
cmake --build build --config Release --target ai_telemetry_tool

echo "===================================================="
echo "                Build Successful!                   "
echo "===================================================="

if [ -z "$1" ]; then
    echo "[!] Warning: No model file provided."
    echo "    Usage: ./build_and_run.sh <path_to_model.gguf>"
    echo "    The executable is available at: ./build/ai_telemetry_tool"
    exit 0
else
    MODEL_PATH="$1"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "[E] Error: Model file not found at '$MODEL_PATH'"
        exit 1
    fi
    
    echo "[*] Launching Telemetry Dashboard with model: $MODEL_PATH"
    echo "===================================================="
    
    if [ -f "./build/Release/ai_telemetry_tool.exe" ]; then
        ./build/Release/ai_telemetry_tool.exe "$MODEL_PATH"
    elif [ -f "./build/ai_telemetry_tool.exe" ]; then
        ./build/ai_telemetry_tool.exe "$MODEL_PATH"
    elif [ -f "./build/ai_telemetry_tool" ]; then
        ./build/ai_telemetry_tool "$MODEL_PATH"
    else
        echo "[E] Error: Could not find the compiled binary artifact inside the build directory."
        exit 1
    fi
fi
