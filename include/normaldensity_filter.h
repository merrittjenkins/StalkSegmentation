#ifndef NORMALDENSITY_FILTER_H
#define NORMALDENSITY_FILTER_H

#include "standard_includes.h"

class normaldensityFilter {

  public:

	normaldensityFilter(const char* filename, const Mat& mat);
	void filter(const Mat& mat, const Mat& img1);
};

#endif
