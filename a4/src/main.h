#ifndef __MAIN_H_
#define __MAIN_H_
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include "libs/bitmap.h"

/* DO NOT OPTIMIZE THIS FILE */

#define colourGradientSteps 7

struct colourGradientStep {
  double step;
  pixel colour;
};


void initColourMap(pixel *colourMap, unsigned int const colourMapSize, struct colourGradientStep const colourGradient[colourGradientSteps])  {
  assert(colourGradient[0].step == 0.0);
  assert(colourGradient[colourGradientSteps - 1].step == 1.0);

  pixel colour = { .r = 0, .g = 0, .b = 0 };
  double pos = 0.0;
  unsigned int ins = 0;

  for (unsigned int i = 0; i < colourGradientSteps && ins < colourMapSize; i++) {
    assert(pos <= colourGradient[i].step);
    int r = (int) colourGradient[i].colour.r - colour.r;
    int g = (int) colourGradient[i].colour.g - colour.g;
    int b = (int) colourGradient[i].colour.b - colour.b;
    unsigned int const max = ceil(((double) colourMapSize * (colourGradient[i].step - pos)));
	for (unsigned int i = 0; i < max && ins < colourMapSize; i++) {
      double blend = (double) i / max;
      colourMap[ins].r = colour.r + (blend * r);
      colourMap[ins].g = colour.g + (blend * g);
      colourMap[ins].b = colour.b + (blend * b);
      ins++;
    }
    pos = colourGradient[i].step;
    colour = colourGradient[i].colour;
  }
}


#endif // __MAIN_H_
