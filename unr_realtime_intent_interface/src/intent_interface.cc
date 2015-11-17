#include "intent_interface.h"
namespace intent {
JPLInterface::JPLInterface() {}
JPLInterface::JPLInterface(std::ofstream *fout) {}
JPLInterface::JPLInterface(uint32_t address, uint32_t port) {}
JPLInterface::~JPLInterface() {}

uint32_t JPLInterface::Start() {}
void JPLInterface::WaitForDone() {}
}  // namespace intent