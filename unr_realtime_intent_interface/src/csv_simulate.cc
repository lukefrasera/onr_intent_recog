#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include "log.h"
#include "intent_interface.h"


int main(int argc, char *argv[]) {
  // Handle command line arguments
  char *filename = NULL;
  float rate = 1.0f;

  while (true) {
    int option_index = 0;
    int c;
    static struct option long_options[] = {
      {"file", required_argument, 0, 0},
      {"rate", required_argument, 0, 0}
    };

    c = getopt_long(argc, argv, "f:r:", long_options, &option_index);

    if (c == -1)
      break;
    const char *name;
    switch (c) {
      case 0:
        name = long_options[option_index].name;
        if (strcmp("file", name) == 0) {
          filename = strdup(optarg);
        } else if (strcmp("rate", name) == 0) {
          rate = atoi(optarg);
        }
        break;
      case 'f':
        if (optarg)
          filename = strdup(optarg);
        break;
      case 'r':
        if (optarg)
          printf("Option r:%s\n", optarg);
        break;
      default:
        printf(
              "Usage: %s [-f <file>] "
              "[-r <rate>]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    if (optind < argc) {
      printf("non-option ARGV-elements: ");
      while (optind < argc)
        printf("%s ", argv[optind++]);
      printf("\n");
    }
  }

  if (filename)
    LOG_INFO("Filename: %s", filename);
  // Initialize file stream
  std::ifstream fin;
  fin.open(filename);

  // Initilialize simulation
  intent::JPLInterface interface(&fin);

  // Start simulation
  interface.Start();
  interface.WaitForDone();

  fin.close();
  if (filename)
    free(filename);
  return 0;
}