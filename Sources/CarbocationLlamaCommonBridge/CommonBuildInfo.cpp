#include "build-info.h"

#include <cstdio>
#include <string>

int llama_build_number(void) {
    return 0;
}

const char * llama_commit(void) {
    return "vendored";
}

const char * llama_compiler(void) {
    return "swiftpm";
}

const char * llama_build_target(void) {
    return "carbocation";
}

const char * llama_build_info(void) {
    static const std::string info = "b0-vendored";
    return info.c_str();
}

void llama_print_build_info(void) {
    std::fprintf(stderr, "%s\n", llama_build_info());
}
