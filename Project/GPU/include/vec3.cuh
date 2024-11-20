#ifndef GPU_VEC3_CUH
#define GPU_VEC3_CUH

// std
#include <cmath>
#include <iostream>

#ifndef datatype
#define datatype float
#endif

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
    __host__ __device__ bool     near_zero() const;
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

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, datatype etai_over_etat) {
    datatype dot_uv = dot(-uv, n);
    datatype cos_theta = dot_uv < datatype(1.0) ? dot_uv : datatype(1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    datatype fabs_val = datatype(1.0) - r_out_perp.length_squared();
    fabs_val = fabs_val < 0 ? fabs_val * datatype(-1.0) : fabs_val;
    vec3 r_out_parallel = -std::sqrt(fabs_val) * n;
    return r_out_perp + r_out_parallel;
}

#endif //GPU_VEC3_CUH