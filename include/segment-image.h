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

#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cstdlib>
#include <image.h>
#include <misc.h>
#include <filter.h>
#include "segment-graph.h"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_map>

// random color
rgb random_rgb()
{
  rgb c;
  double r;

  c.r = (uchar)random();
  c.g = (uchar)random();
  c.b = (uchar)random();

  return c;
}

// dissimilarity measure between pixels
static inline float diff(image<float> *r, image<float> *g, image<float> *b,
                         int x1, int y1, int x2, int y2)
{
  return sqrt(square(imRef(r, x1, y1) - imRef(r, x2, y2)) +
              square(imRef(g, x1, y1) - imRef(g, x2, y2)) +
              square(imRef(b, x1, y1) - imRef(b, x2, y2)));
}

/**
 * Method that isolates each single segment
 * @author Paolo Galletta
 * @param u the segmentation graph
 * @param width image width
 * @param height image height
 * @return the vector containing the mask of each single segment
 **/
std::vector<cv::Mat> get_segment_mask(universe *u, int width, int height)
{
  std::vector<cv::Mat> output;
  int num_sets = u->num_sets();

  // To isolate the segments, an hash map is used
  std::unordered_multimap<int, cv::Point> hash_map;

  // Traverse the image
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int comp = u->find(y * width + x);

      // Group the points belonging to the same segment
      hash_map.insert({comp, cv::Point(x, y)});
    }
  }

  // Create the Mat single masks
  for (auto it = hash_map.begin(); it != hash_map.end();)
  {
    cv::Mat current_segment(cv::Size(width, height), CV_8UC1);
    auto const &key = it->first;

    auto range = hash_map.equal_range(key);

    auto iter = range.first;
    for (; iter != range.second; ++iter)
    {
      current_segment.at<uchar>(iter->second) = 255;
    }
    output.push_back(current_segment);

    while (++it != hash_map.end() && it->first == key)
      ;
  }
  return output;
}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */

image<rgb> *segment_image(image<rgb> *im, float sigma, float c, int min_size,
                          int *num_ccs, std::vector<cv::Mat> *masks)
{
  int width = im->width();
  int height = im->height();

  image<float> *r = new image<float>(width, height);
  image<float> *g = new image<float>(width, height);
  image<float> *b = new image<float>(width, height);

  // smooth each color channel
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      imRef(r, x, y) = imRef(im, x, y).r;
      imRef(g, x, y) = imRef(im, x, y).g;
      imRef(b, x, y) = imRef(im, x, y).b;
    }
  }
  image<float> *smooth_r = smooth(r, sigma);
  image<float> *smooth_g = smooth(g, sigma);
  image<float> *smooth_b = smooth(b, sigma);
  delete r;
  delete g;
  delete b;

  // build graph
  edge *edges = new edge[width * height * 4];
  int num = 0;
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      if (x < width - 1)
      {
        edges[num].a = y * width + x;
        edges[num].b = y * width + (x + 1);
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y);
        num++;
      }

      if (y < height - 1)
      {
        edges[num].a = y * width + x;
        edges[num].b = (y + 1) * width + x;
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y + 1);
        num++;
      }

      if ((x < width - 1) && (y < height - 1))
      {
        edges[num].a = y * width + x;
        edges[num].b = (y + 1) * width + (x + 1);
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y + 1);
        num++;
      }

      if ((x < width - 1) && (y > 0))
      {
        edges[num].a = y * width + x;
        edges[num].b = (y - 1) * width + (x + 1);
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y - 1);
        num++;
      }
    }
  }
  delete smooth_r;
  delete smooth_g;
  delete smooth_b;

  // segment
  universe *u = segment_graph(width * height, num, edges, c);

  // post process small components
  for (int i = 0; i < num; i++)
  {
    int a = u->find(edges[i].a);
    int b = u->find(edges[i].b);
    if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
      u->join(a, b);
  }
  delete[] edges;
  *num_ccs = u->num_sets();
  *masks = get_segment_mask(u, width, height);
  image<rgb> *output = new image<rgb>(width, height);

  // pick random colors for each component
  rgb *colors = new rgb[width * height];
  for (int i = 0; i < width * height; i++)
    colors[i] = random_rgb();

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int comp = u->find(y * width + x);
      imRef(output, x, y) = colors[comp];
    }
  }

  delete[] colors;
  delete u;

  return output;
}

#endif
