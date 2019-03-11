#include "hfststringstream_cpp.h"

hfst::HfstInputStream HfstInputStringStream(std::string &str) {
  std::istringstream is(str);
  return hfst::HfstInputStream(is);
}

