#!/bin/bash

set -e

MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_MODEL="qwen2.5-1.5b-instruct-q4_k_m.gguf"

echo "===================================================="
echo "    Local LLM Instrumentation & Telemetry Platform  "
echo "===================================================="

if [ -z "$1" ]; then
    MODEL_PATH="$DEFAULT_MODEL"
    echo "[*] No custom model provided. Defaulting to: $DEFAULT_MODEL"
else
    MODEL_PATH="$1"
fi

if [ ! -f "$MODEL_PATH" ]; then
    if [ "$MODEL_PATH" == "$DEFAULT_MODEL" ]; then
        echo "[!] Default model not found in the current directory."
        echo "[*] Bootstrapping: Automatically downloading Qwen 2.5 (1.1 GB) for telemetry testing..."
        echo "[*] (This will only happen once. Please wait...)"
        # Uses curl to follow redirects (-L) and save with the remote name (-O)
        curl -L -O "$MODEL_URL"
        echo "[*] Download complete!"
    else
        echo "[E] Error: Custom model file not found at '$MODEL_PATH'"
        echo "    Please check your path or run without arguments to auto-download the default model."
        exit 1
    fi
fi

echo "----------------------------------------------------"
echo "[*] Compiling C++ Telemetry Engine..."
echo "----------------------------------------------------"

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target ai_telemetry_tool

echo "===================================================="
echo "      Build Successful! Launching TUI Dashboard...  "
echo "===================================================="

if [ -f "./build/bin/Release/ai_telemetry_tool.exe" ]; then
    ./build/bin/Release/ai_telemetry_tool.exe "$MODEL_PATH"
elif [ -f "./build/Release/ai_telemetry_tool.exe" ]; then
    ./build/Release/ai_telemetry_tool.exe "$MODEL_PATH"
elif [ -f "./build/ai_telemetry_tool.exe" ]; then
    ./build/ai_telemetry_tool.exe "$MODEL_PATH"
elif [ -f "./build/ai_telemetry_tool" ]; then
    ./build/ai_telemetry_tool "$MODEL_PATH"
else
    echo "[E] Fatal Error: Could not locate the compiled executable."
    exit 1
fi
