#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <iostream>

using namespace ftxui;

int main() {
    // Create a simple box with text inside
    Element document = vbox({
        text("FTXUI Successfully Downloaded and Linked!") | bold,
        text("Ready to build the telemetry UI.")
    }) | border;

    // Render to terminal
    auto screen = Screen::Create(Dimension::Fit(document), Dimension::Fit(document));
    Render(screen, document);
    screen.Print();

    return 0;
}