#include <gtest/gtest.h>

#include "Util.h"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  initCL_nvidia();
  int retval = RUN_ALL_TESTS();
  freeCL();

  return retval;
}
