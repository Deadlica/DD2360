#ifndef GPU_VEC3_CUH
#define GPU_VEC3_CUH

// Project
#include <util.cuh>

// std
#include <cmath>
#include <iostream>

class vec3 {
public:
    datatype e[3];

    __host__ __device__ vec3();
    __host__ __device__ vec3(datatype e0, datatype e1, datatype e2);

    __host__ __device__ datatype x() const;
    __host__ __device__ datatype y() const;
    __host__ __device__ datatype z() const;
    __host__ __device__ datatype r() const;
    __host__ __device__ datatype g() const;
    __host__ __device__ datatype b() const;

    __host__ __device__ const vec3& operator+() const;
    __host__ __device__ vec3 operator-() const;
    __host__ __device__ datatype operator[](int i) const;
    __host__ __device__ datatype& operator[](int i);

    __host__ __device__ vec3& operator+=(const vec3& rhs);
    __host__ __device__ vec3& operator-=(const vec3& rhs);
    __host__ __device__ vec3& operator*=(const vec3& rhs);
    __host__ __device__ vec3& operator/=(const vec3& rhs);
    __host__ __device__ vec3& operator*=(datatype rhs);
    __host__ __device__ vec3& operator/=(datatype rhs);

    __host__ __device__ datatype length() const;
    __host__ __device__ datatype length_squared() const;
    __host__ __device__ void     make_unit_vector();
};

using point3 = vec3;

inline std::istream& operator>>(std::istream& in, vec3& v) {
    return in >> v.e[0] >> v.e[1] >> v.e[2];
}

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << " " << v.e[1] << " " << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] + rhs.e[0], lhs.e[1] + rhs.e[1], lhs.e[2] + rhs.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] - rhs.e[0], lhs.e[1] - rhs.e[1], lhs.e[2] - rhs.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] * rhs.e[0], lhs.e[1] * rhs.e[1], lhs.e[2] * rhs.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] / rhs.e[0], lhs.e[1] / rhs.e[1], lhs.e[2] / rhs.e[2]);
}

__host__ __device__ inline vec3 operator*(datatype lhs, const vec3& rhs) {
    return vec3(lhs * rhs.e[0], lhs * rhs.e[1], lhs * rhs.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& lhs, datatype rhs) {
    return (1 / rhs) * lhs;
}

__host__ __device__ inline vec3 operator*(const vec3& lhs, datatype rhs) {
    return rhs * lhs;
}

__host__ __device__ inline datatype dot(const vec3& lhs, const vec3& rhs) {
    return lhs.e[0] * rhs.e[0]
           + lhs.e[1] * rhs.e[1]
           + lhs.e[2] * rhs.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[1] * rhs.e[2] - lhs.e[2] * rhs.e[1],
                lhs.e[2] * rhs.e[0] - lhs.e[0] * rhs.e[2],
                lhs.e[0] * rhs.e[1] - lhs.e[1] * rhs.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

#endif //GPU_VEC3_CUH