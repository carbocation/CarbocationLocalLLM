#ifdef DEBUG
#undef DEBUG
#endif

// SwiftPM defines DEBUG=1 for debug C/C++ builds, while upstream mtmd-audio.cpp
// declares a local constexpr named DEBUG. Include it through this wrapper so the
// target can build without unsafe -UDEBUG flags.
#include "../../Vendor/llama.cpp/tools/mtmd/mtmd-audio.cpp"
