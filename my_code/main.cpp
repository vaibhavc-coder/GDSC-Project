#include "llama.h"
#include "llama-telemetry.h" 
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.gguf>\n";
        return 1;
    }

    TelemetryEngine engine;

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(argv[1], model_params);

    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.cb_eval = ggml_eval_callback; 
    ctx_params.cb_eval_user_data = &engine;
    ctx_params.n_ctx = 2048; 

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    std::string prompt = "Explain the theory of relativity briefly: ";
    std::vector<llama_token> tokens(prompt.length() + 4);
    int n_tokens = llama_tokenize(model, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(model, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    auto sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add_greedy(sampler);
    
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    int max_tokens_to_generate = 30; 
    
    for (int i = 0; i < max_tokens_to_generate; i++) {
        
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_token_is_eog(model, new_token)) {
            break; 
        }
        batch = llama_batch_get_one(&new_token, 1);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
    }

    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
