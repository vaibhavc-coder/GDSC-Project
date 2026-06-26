#!/bin/bash

echo "=== 1. Initializing Submodules ==="
git submodule update --init --recursive

echo "=== 2. Injecting Telemetry Hook ==="
cp my_code/llama-telemetry.h external/llama.cpp/src/
if ! grep -q "TelemetryEngine" external/llama.cpp/src/llama-context.cpp; then
    patch external/llama.cpp/src/llama-context.cpp < my_code/llama-context.patch
fi

echo "=== 3. Compiling from Root ==="
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

echo "=== 4. Compilation Complete! ==="
echo "The executable is now located in your root build folder."
echo "To test the telemetry dashboard, run:"
echo "./build/submodules/llama.cpp/bin/llama-cli -m <path_to_your_model> -p 'Hello' -n 50"
