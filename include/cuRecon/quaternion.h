#include "types.h"
#include <math.h>

quaternion quaternion_from_axis_angle(float axisX, float axisY, float axisZ, float angle);
quaternion normalize(const quaternion &q);
quaternion conjugate(const quaternion &q);
quaternion multiply(const quaternion &q1, const quaternion &q2);
point rotate_point(const quaternion &q, const point &v);
pointcloud rotate_pointcloud(const pointcloud &pointCloud, const quaternion &rotation);
quaternion identity_quaternion();
