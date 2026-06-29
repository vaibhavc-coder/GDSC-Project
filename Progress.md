->installed cmake and completed FTXUI setup 
->explored FTXUI examples and components from git repo
->built a basic terminal layout for the project
->started learning how transformers work (attention,tensors,mlp,embedding etc.)
->Ran a LLM in the terminal by using the llama-cli / llama-server binary as-is
<img width="1750" height="1622" alt="Screenshot 2026-06-19 115859" src="https://github.com/user-attachments/assets/581e3aeb-6685-4811-80ae-ac409c7cca8b" />
->Ran a LLM locally by cloning llama.cpp and build it myself
<img width="2858" height="1600" alt="Screenshot 2026-06-19 132013" src="https://github.com/user-attachments/assets/0ba297dd-55d7-4a55-80c2-b66d9c44dc8a" />
Add llama.cpp and ftxui submodule at external
->hook non-invasively using ggml_eval_callback pointer in main.cpp
->build llama-telemetry.h for tui dashboard
->compile code and fix bugs
->made build_and_run.sh for easy reviewing
