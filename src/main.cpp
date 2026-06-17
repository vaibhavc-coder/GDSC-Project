#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp> 

#include <mutex>
#include <deque>
#include <string>
#include <vector> 

using namespace ftxui;

struct LayerMetrics {
    std::string name;
    float latency_ms;
    std::vector<int> tensor_shape;
    float sparsity_rate;
};

struct AppState {
    std::mutex mtx;
    std::deque<LayerMetrics> layer_buffer; 
    int max_buffer_size = 100;
    
    int active_tab = 0;  
    int selected_layer = 0; 
};

int main() {
    auto screen = ScreenInteractive::Fullscreen();
    AppState state;

    std::vector<std::string> tab_entries = {
        "Overview", "Attention Matrix", "Anomalies"
    };
    auto tab_selection = Toggle(&tab_entries, &state.active_tab);

    std::vector<std::string> layer_names = {
        "Embedding", "Block 0", "Block 1", "Block 2", "LM Head"
    };
    
    auto base_menu = Menu(&layer_names, &state.selected_layer);

    auto vim_layer_menu = CatchEvent(base_menu, [&](Event event) {
        if (event == Event::Character('j')) {
            if (state.selected_layer < (int)layer_names.size() - 1) state.selected_layer++;
            return true; 
        }
        if (event == Event::Character('k')) {
            if (state.selected_layer > 0) state.selected_layer--;
            return true; 
        }
        if (event == Event::Tab) {
            state.active_tab = (state.active_tab + 1) % tab_entries.size();
            return true;
        }
        return false; 
    });

    auto overview_render = Renderer([&] {
        return vbox({
            text("Latency: 12.4ms") | bold,
            text("Shape: [1, 32, 4096]"),
            text("Sparsity: 15%") | color(Color::Green),
        });
    });

    auto attention_render = Renderer([&] {
        return text("Attention Matrix Visualization goes here...");
    });

    auto anomalies_render = Renderer([&] {
        return text("No clipping risks detected.");
    });

    auto tab_content = Container::Tab({
        overview_render,
        attention_render,
        anomalies_render
    }, &state.active_tab);

    auto main_container = Container::Horizontal({
        vim_layer_menu,
        tab_content
    });

    auto layout = Container::Vertical({
        tab_selection,
        main_container
    });

    auto final_render = Renderer(layout, [&] {
        return vbox({
            text(" Transformer Telemetry Dashboard ") | bold | center | color(Color::Red),
            text("By Vaibhav and Om Kiran") | align_right,
            hbox({
                text("[Q]") | color(Color::RedLight),
                text(": Quit App"),
                separator(),
                text("[Tab]") | color(Color::RedLight),
                text(": Tab movement"),
                separator(),
                text("[j/k]") | color(Color::RedLight),
                text(": layer movement"),
            }) | center,
            separator(),
            tab_selection->Render() | center,
            separator(),
            hbox({
                window(text(" Layers ") | color(Color::CyanLight), vim_layer_menu->Render()) | size(WIDTH, EQUAL, 25),
                window(text(" Details ") | color(Color::CyanLight), tab_content->Render()) | flex,
            }) | flex
        });
    });

    screen.Loop(final_render);
    return 0;
}
