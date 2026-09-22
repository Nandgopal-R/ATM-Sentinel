// Host harness: reads a 300x8 CSV window (empty cell = null) on stdin, prints
// fault mask + the 57 features. Driven by test_features.py.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../atm_sentinel/features.h"
int main() {
  static float lin[N_CHAN][WINDOW_S];
  char line[512];
  for (int i = 0; i < WINDOW_S; i++) {
    if (!fgets(line, sizeof line, stdin)) return 1;
    char* p = line;
    for (int c = 0; c < N_CHAN; c++) {
      char* e = strpbrk(p, ",\n");
      if (e) *e = 0;
      lin[c][i] = *p ? strtof(p, nullptr) : NAN;
      p = e ? e + 1 : p + strlen(p);
    }
  }
  const float* chan[N_CHAN];
  for (int c = 0; c < N_CHAN; c++) chan[c] = lin[c];
  float feat[N_FEATURES];
  unsigned mask = extractFeatures(chan, feat, 60);
  printf("%u", mask);
  for (int i = 0; i < N_FEATURES; i++) printf(",%.9g", feat[i]);
  printf("\n");
}
