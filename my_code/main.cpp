#include "llama.h"
#include "llama-telemetry.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.gguf>\n";
        return 1;
    }

    TelemetryEngine engine;

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(argv[1], model_params);

    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.cb_eval = ggml_eval_callback; 
    ctx_params.cb_eval_user_data = &engine; 

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    std::string prompt = "Explain the theory of relativity in one sentence.";
    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
