# Local LLM Telemetry Dashboard
By: Vaibhav Chekka

# 1. Overview & Features
This is a C++ terminal tool that lets you look inside a Large Language Model (LLM) while it generates text. Instead of treating the AI like a black box, this dashboard hooks into the model's forward pass to show you exactly what is happening layer by layer.

It tracks how long each block takes to run, calculates sparsity, and draws a live heatmap of the tensor activations—all directly in your terminal.

# Key Features
Zero Code Changes to the Model: It uses the ggml_eval_callback pointer to read memory on the fly. I didn't have to alter any of the core llama.cpp code to get this working.

Interactive Terminal UI: You can navigate the dashboard using j/k keys, press Space to lock onto a specific layer, and use Tab to switch over to the Anomaly Ledger.

Live Heatmaps: Draws an 8x8 representation of the Attention and FFN matrices as tokens pass through them.

Anomaly Tracker (Bonus Feature): Automatically flags any numerical spikes (like values over 15.0 or NaNs) to warn about clipping risks.

Auto-Setup Script: Includes a shell script that handles everything for the reviewer in one click.

# 2. Quickstart: Compile and Run
This project includes an automated bootstrapping script designed for zero-friction reviewer testing.

Prerequisites
:- g++ or MSVC (C++17 or higher)

:- CMake

:- Git / Git Bash

Execution
Simply open your terminal in the root project directory and run the following command:

: Bash
: ./build_and_run.sh
What this script does automatically:

Detects if a compatible .gguf model is present.

If no model is found, it automatically downloads Qwen 2.5 1.5B Instruct (~1.1 GB) via curl.

Purges any stale CMake cache and compiles the C++ telemetry engine statically.

Launches the TUI Dashboard.

# Controls
j / k : Move up and down the lists.

Space : Pause/Lock the screen on a specific layer so you can read its stats. (The AI keeps generating tokens in the background).

Tab : Switch between the left panel and the Anomaly Ledger.

q : Quit the program.

# 3. Notes for the Reviewer
While building this, I ran into a few interesting architectural issues that I had to work around to get it running smoothly:

## 1. Reverting the llama.cpp Engine
If you look at the Git history, you'll see I locked the llama.cpp library to an older version (Release Tag b3800). The absolute newest version of llama.cpp recently updated how its memory scheduler works, which currently crashes the system if you try to pause the pipeline and read the tensors. Using this stable older tag safely bypasses that bug.

## 2. Background Math Processing
Instead of forcing the AI to stop completely while the tool calculates sparsity and finds the max activation values, I moved all that heavy math to a separate background thread (using a Producer-Consumer queue). The AI just hands off a quick copy of the memory and keeps running, which keeps the token generation fast.

## 3. Why Qwen 2.5?
The script downloads a Qwen model by default instead of something newer like Gemma 3. This is because Gemma 3 uses a hybrid State-Space architecture that drops a lot of the standard Attention/FFN layers. Qwen is a pure, classic Transformer, which makes it the perfect model to show off the layer heatmaps for this assignment.
