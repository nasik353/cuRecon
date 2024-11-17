#ifndef TYPES_H
#define TYPES_H

#include <vector>

struct quaternion
{
    float x;
    float y;
    float z;
    float w;
};

struct image
{
    int width;
    int height;
    std::vector<unsigned char> raw_data;
};

struct point
{
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;
};

struct pointcloud
{
    std::vector<point> data;
};

struct frame
{
    int _id;
    image _image;
    pointcloud _pointcloud;
    pointcloud _downsampled_pointcloud;

    quaternion _q_init;
    quaternion _q_trans;

    bool _last;
};

#endif // TYPES_H