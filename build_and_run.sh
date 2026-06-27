#!/bin/bash

set -e

echo "===================================================="
echo " Starting Non-Invasive AI Telemetry Platform Build "
echo "===================================================="

cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF
cmake --build build --config Release --target ai_telemetry_tool

echo "===================================================="
echo "                Build Successful!                   "
echo "===================================================="

if [ -z "$1" ]; then
    exit 0
else
    MODEL_PATH="$1"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "[E] Error: Model file not found at $MODEL_PATH"
        exit 1
    fi
    
    if [ -f "./build/bin/Release/ai_telemetry_tool.exe" ]; then
        ./build/bin/Release/ai_telemetry_tool.exe "$MODEL_PATH"
    elif [ -f "./build/Release/ai_telemetry_tool.exe" ]; then
        ./build/Release/ai_telemetry_tool.exe "$MODEL_PATH"
    elif [ -f "./build/ai_telemetry_tool.exe" ]; then
        ./build/ai_telemetry_tool.exe "$MODEL_PATH"
    elif [ -f "./build/ai_telemetry_tool" ]; then
        ./build/ai_telemetry_tool "$MODEL_PATH"
    else
        echo "[E] Error: Could not find the compiled binary."
        exit 1
    fi
fi
