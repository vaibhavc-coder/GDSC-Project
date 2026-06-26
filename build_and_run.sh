#!/bin/bash
set -e

echo "=== 1. Initializing Submodules ==="
git submodule update --init --recursive

echo "=== 2. Injecting Telemetry Hook ==="
cp my_code/llama-telemetry.h external/llama.cpp/src/
cp my_code/llama-context.cpp external/llama.cpp/src/

echo "=== 3. Compiling from Root ==="
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --config Release -j 8

echo "=== 4. Compilation Complete! ==="
echo "Locate llama-cli with:"
echo "  find build -iname 'llama-cli*' -type f"
echo ""
echo "Then run it, e.g.:"
echo "  ./build/bin/llama-cli -m <path_to_your_model> -p 'Hello' -n 50"
echo "(exact path depends on your CMake generator - see the find command above if this path is wrong)"
