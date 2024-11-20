// Project
#include <vec3.cuh>

__host__ __device__ vec3::vec3(): e{0, 0, 0} {}

__host__ __device__ vec3::vec3(datatype e0, datatype e1, datatype e2): e{e0, e1, e2} {}

__host__ __device__ datatype vec3::x() const {
    return e[0];
}

__host__ __device__ datatype vec3::y() const {
    return e[1];
}

__host__ __device__ datatype vec3::z() const {
    return e[2];
}

__host__ __device__ datatype vec3::r() const {
    return e[0];
}

__host__ __device__ datatype vec3::g() const {
    return e[1];
}

__host__ __device__ datatype vec3::b() const {
    return e[2];
}

__host__ __device__ const vec3& vec3::operator+() const {
    return *this;
}

__host__ __device__ vec3 vec3::operator-() const {
    return vec3(-e[0], -e[1], -e[2]);
}

__host__ __device__ datatype vec3::operator[](int i) const {
    return e[i];
}

__host__ __device__ datatype& vec3::operator[](int i) {
    return e[i];
}

__host__ __device__ vec3& vec3::operator+=(const vec3& rhs) {
    e[0] += rhs.e[0];
    e[1] += rhs.e[1];
    e[2] += rhs.e[2];
    return *this;
}

__host__ __device__ vec3& vec3::operator-=(const vec3& rhs) {
    e[0] -= rhs.e[0];
    e[1] -= rhs.e[1];
    e[2] -= rhs.e[2];
    return *this;
}

__host__ __device__ vec3& vec3::operator*=(const vec3& rhs) {
    e[0] *= rhs.e[0];
    e[1] *= rhs.e[1];
    e[2] *= rhs.e[2];
    return *this;
}

__host__ __device__ vec3& vec3::operator/=(const vec3& rhs) {
    e[0] /= rhs.e[0];
    e[1] /= rhs.e[1];
    e[2] /= rhs.e[2];
    return *this;
}

__host__ __device__ vec3& vec3::operator*=(datatype rhs) {
    e[0] *= rhs;
    e[1] *= rhs;
    e[2] *= rhs;
    return *this;
}

__host__ __device__ vec3& vec3::operator/=(datatype rhs) {
    return *this *= 1 / rhs;
}

__host__ __device__ datatype vec3::length() const {
    return std::sqrt(length_squared());
}

__host__ __device__ datatype vec3::length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}

__host__ __device__ bool vec3::near_zero() const {
    datatype s = 1e-8;
    return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
}

__host__ __device__ void vec3::make_unit_vector() {
    datatype k = 1.0 / std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}
