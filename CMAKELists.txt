cmake_minimum_required(VERSION 3.14)
project(ai_telemetry_tool)

include(FetchContent)
FetchContent_Declare(ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/ftxui
  GIT_TAG v5.0.0
)
FetchContent_MakeAvailable(ftxui)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable llama.cpp tests" FORCE)
set(LLAMA_BUILD_EXAMPLES ON CACHE BOOL "Build llama.cpp examples (like llama-cli)" FORCE)

add_subdirectory(external/llama.cpp)

target_link_libraries(llama PUBLIC 
    ftxui::screen
    ftxui::dom
    ftxui::component
)
