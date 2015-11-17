#ifndef INTENT_INTERFACE_H_
#define INTENT_INTERFACE_H_
#include <stdint.h>
#include <thread>
#include <sys/socket.h>
#include "intent.pb.h"

namespace intent {
typedef struct JPLMsgOut {
  double    timestamp;
  uint16_t  source_id;
  uint16_t  contact_id;
  uint16_t  contact_type;
  uint16_t  intent_type;
  uint16_t  group_id;
  uint16_t  intent_class;
  float     intent_prob;
  float     group_confidence;
} JPLMsgOut_t;

// class JPLInterface {
//  public:
//   JPLInterface();
//   JPLInterface(ifstream *fin);
//   JPLInterface(AF_INET address, uint32_t port);

//   virtual ~JPLInterface();

//   uint32_t Start();
//   void WaitForDone();

//  private:
//   std::thread *main_thread;
// };
}
#endif  // INTENT_INTERFACE_H_
