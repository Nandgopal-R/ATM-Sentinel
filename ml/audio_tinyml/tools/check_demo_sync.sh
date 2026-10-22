#!/bin/sh
# The demo sketch keeps its own copy of the module files (Arduino compiles only its own folder).
# firmware/atm_sentinel/ is the source of truth; fail if the demo copy has drifted from it.
cd "$(dirname "$0")/../../../firmware" || exit 1
rc=0
for f in esp32_preprocessing.h esp32_preprocessing.cpp esp32_inference.h esp32_inference.cpp \
         atm_model_data.h test_vectors.h; do
  cmp -s "atm_sentinel/$f" "tests/atm_tinyml_demo/$f" ||
    { echo "DIFFERS: $f  (copy firmware/atm_sentinel/$f into firmware/tests/atm_tinyml_demo/)"; rc=1; }
done
[ $rc = 0 ] && echo "demo sketch in sync with firmware/atm_sentinel (6 files)"
exit $rc
