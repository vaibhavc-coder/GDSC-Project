#include <memory>  
#include <string> 
#include <vector> 

#include "ftxui/component/app.hpp"         
#include "ftxui/component/captured_mouse.hpp"
#include "ftxui/component/component.hpp"
#include "ftxui/component/component_base.hpp"  
#include "ftxui/dom/elements.hpp"  

using namespace ftxui;

int main() {
  std::vector<std::string> tab_values{
      "1. MODEL TOPOLOGY (Focus Active)",
      "2. LIVE PACKET STREAM",
      "3. ATTENTION MATRIX VISUALIZER (HEAD 0)",
      "4. RUNTIME METRICS INSPECTOR",
      "5. NUMERICAL ANOMALY LEDGER",
  };
  int tab_selected = 0;
  auto tab_toggle = Toggle(&tab_values, &tab_selected);

  std::vector<std::string> tab_1_entries{
      "To be filled",
  };
  int tab_1_selected = 0;

  std::vector<std::string> tab_2_entries{
      "To be filled",
  };
  int tab_2_selected = 0;

  std::vector<std::string> tab_3_entries{
      "To be filled",
  };
  int tab_3_selected = 0;

  std::vector<std::string> tab_4_entries{
      "To be filled",
  };
  int tab_4_selected = 0;

  std::vector<std::string> tab_5_entries{
      "To be filled",
  };
  int tab_5_selected = 0;

  auto tab_container = Container::Tab(
      {
          Radiobox(&tab_1_entries, &tab_1_selected),
          Radiobox(&tab_2_entries, &tab_2_selected),
          Radiobox(&tab_3_entries, &tab_3_selected),
          Radiobox(&tab_4_entries, &tab_4_selected),
          Radiobox(&tab_5_entries, &tab_5_selected),
      },
      &tab_selected);

  auto container = Container::Vertical({
      tab_toggle,
      tab_container,
  });

  auto renderer = Renderer(container, [&] {
    return vbox({
               tab_toggle->Render(),
               separator(),
               tab_container->Render(),
           }) |
           border;
  });

  auto screen = App::TerminalOutput();
  screen.Loop(renderer);
}
