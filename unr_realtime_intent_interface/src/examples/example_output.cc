#include "intent_interface.h"
#include "log.h"
#include <stdio.h>
#include <boost/date_time.hpp>
#include <boost/thread.hpp>

intent::JPLMsgOut_t GetRandomJPMsgWithTime() {
  intent::JPLMsgOut_t data;
  boost::posix_time::ptime time_t_epoch(boost::gregorian::date(1970,1,1));
  boost::posix_time::time_duration diff = boost::posix_time::microsec_clock::universal_time() - time_t_epoch;

  data.timestamp = (double)diff.total_seconds() + (double)diff.fractional_seconds() / 1000000.0;
  data.source_id = 1;
  data.contact_id = 4;
  data.contact_type = 99;
  data.intent_type = 1;
  data.group_id = 1;
  data.intent_class = 1;
  data.intent_prob = .66;
  data.group_confidence = .88;
  return data;
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    printf("Error: filename is required\n");
    return -1;
  }
  FILE *fout;
  fout = fopen(argv[1], "w");

  fprintf(fout, "timestamp, source_id, contact_id, contact_type, intent_type, group_id, intent_class, intent_prob, group_confidence\n");
  intent::JPLMsgOut_t data;
  for (int i = 0; i < 100; ++i) {
    data = GetRandomJPMsgWithTime();
    fprintf(fout, "%f, %d, %d, %d, %d, %d, %d, %f, %f\n",
                                                      data.timestamp,
                                                      data.source_id,
                                                      data.contact_id,
                                                      data.contact_type,
                                                      data.intent_type,
                                                      data.group_id,
                                                      data.intent_class,
                                                      data.intent_prob,
                                                      data.group_confidence);
    LOG_INFO(fout, "%f, %d, %d, %d, %d, %d, %d, %f, %f",
                                                      data.timestamp,
                                                      data.source_id,
                                                      data.contact_id,
                                                      data.contact_type,
                                                      data.intent_type,
                                                      data.group_id,
                                                      data.intent_class,
                                                      data.intent_prob,
                                                      data.group_confidence);
    boost::this_thread::sleep(boost::posix_time::millisec(100));
  }
  fclose(fout);
  return 0;
}