/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/* some image utilities */

#ifndef IMUTIL_H
#define IMUTIL_H

#include "image.h"
#include "misc.h"
#include <opencv2/core.hpp>

/* compute minimum and maximum value in an image */
template <class T>
void min_max(image<T> *im, T *ret_min, T *ret_max)
{
  int width = im->width();
  int height = im->height();

  T min = imRef(im, 0, 0);
  T max = imRef(im, 0, 0);
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      T val = imRef(im, x, y);
      if (min > val)
        min = val;
      if (max < val)
        max = val;
    }
  }

  *ret_min = min;
  *ret_max = max;
}

/* threshold image */
template <class T>
image<uchar> *threshold(image<T> *src, int t)
{
  int width = src->width();
  int height = src->height();
  image<uchar> *dst = new image<uchar>(width, height);

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      imRef(dst, x, y) = (imRef(src, x, y) >= t);
    }
  }

  return dst;
}

/* convert image to Mat */
cv::Mat convertNativeToMat(image<rgb> *input)
{
  int w = input->width();
  int h = input->height();
  cv::Mat output(cv::Size(w, h), CV_8UC3);

  for (int i = 0; i < h; i++)
  {
    for (int j = 0; j < w; j++)
    {
      rgb curr = input->data[i * w + j];
      output.at<cv::Vec3b>(i, j)[0] = curr.b;
      output.at<cv::Vec3b>(i, j)[1] = curr.g;
      output.at<cv::Vec3b>(i, j)[2] = curr.r;
    }
  }

  return output;
}
#endif
