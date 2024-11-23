#include "cuRecon/quaternion.h"

quaternion quaternion_from_axis_angle(float axisX, float axisY, float axisZ, float angle)
{
    quaternion q;
    float halfAngle = angle * 0.5f;
    float sinHalfAngle = std::sin(halfAngle);
    q.x = axisX * sinHalfAngle;
    q.y = axisY * sinHalfAngle;
    q.z = axisZ * sinHalfAngle;
    q.w = std::cos(halfAngle);
    return q;
}

quaternion normalize(const quaternion &q)
{
    float length = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (length == 0)
        return {0, 0, 0, 1};
    return {q.x / length, q.y / length, q.z / length, q.w / length};
}

quaternion conjugate(const quaternion &q) { return {-q.x, -q.y, -q.z, q.w}; }

quaternion multiply(const quaternion &q1, const quaternion &q2)
{
    return {
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y, q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w, q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z};
}

point rotate_point(const quaternion &q, const point &v)
{
    quaternion qVec = {v.x, v.y, v.z, 0};
    quaternion qConjugate = conjugate(q);
    quaternion result = multiply(multiply(q, qVec), qConjugate);
    return {result.x, result.y, result.z, v.r, v.g, v.b};
}

pointcloud rotate_pointcloud(const pointcloud &pointCloud, const quaternion &rotation)
{
    pointcloud rotatedPointCloud;
    rotatedPointCloud.data.reserve(pointCloud.data.size());

    quaternion normalizedRotation = normalize(rotation);
    for (const auto &point : pointCloud.data)
    {
        rotatedPointCloud.data.push_back(rotate_point(normalizedRotation, point));
    }

    return rotatedPointCloud;
}

quaternion identity_quaternion() { return {0, 0, 0, 1}; }
