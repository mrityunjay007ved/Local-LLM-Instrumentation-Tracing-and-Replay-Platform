cmake_minimum_required(VERSION 3.14)
project(transformer-telemetry VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── llama.cpp path ────────────────────────────────────────
# Pass with: cmake -B build -DLLAMA_DIR=/path/to/llama.cpp
# Default assumes it's a sibling folder
if(NOT DEFINED LLAMA_DIR)
    set(LLAMA_DIR "${CMAKE_SOURCE_DIR}/../llama.cpp")
endif()

message(STATUS "Using llama.cpp at: ${LLAMA_DIR}")

# verify llama.cpp exists
if(NOT EXISTS "${LLAMA_DIR}/include/llama.h")
    message(FATAL_ERROR
        "llama.h not found at ${LLAMA_DIR}/include/llama.h\n"
        "Set the correct path with: cmake -B build -DLLAMA_DIR=/path/to/llama.cpp"
    )
endif()

# ── FTXUI (fetched automatically) ────────────────────────
include(FetchContent)
FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
    GIT_TAG        v6.0.2
    GIT_SHALLOW    TRUE  # faster clone
)
set(FTXUI_BUILD_DOCS    OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ftxui)

# ── main executable ───────────────────────────────────────
add_executable(llama-telemetry src/telemetry.cpp)

target_include_directories(llama-telemetry PRIVATE
    ${CMAKE_SOURCE_DIR}/include          # ring_buffer.h
    ${LLAMA_DIR}/include                 # llama.h
    ${LLAMA_DIR}/ggml/include            # ggml.h
)

target_link_directories(llama-telemetry PRIVATE
    ${LLAMA_DIR}/build/src               # libllama
    ${LLAMA_DIR}/build/ggml/src          # libggml
    ${LLAMA_DIR}/build/ggml/src/ggml-cpu # libggml-cpu
)

target_link_libraries(llama-telemetry PRIVATE
    llama
    ggml
    ftxui::screen
    ftxui::dom
    ftxui::component
)

# macOS needs these frameworks for Metal
if(APPLE)
    target_link_libraries(llama-telemetry PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
        "-framework Accelerate"
    )
endif()

target_compile_features(llama-telemetry PRIVATE cxx_std_17)

# ── build instructions printed after cmake configures ────
message(STATUS "")
message(STATUS "Build with:")
message(STATUS "  cmake --build build -j4")
message(STATUS "Run with:")
message(STATUS "  ./build/llama-telemetry model.gguf 'your prompt'")
message(STATUS "")