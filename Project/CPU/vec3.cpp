#include "vec3.h"

vec3::vec3(): e{0, 0, 0} {}

vec3::vec3(datatype e0, datatype e1, datatype e2): e{e0, e1, e2} {}

datatype vec3::x() const {
    return e[0];
}

datatype vec3::y() const {
    return e[1];
}

datatype vec3::z() const {
    return e[2];
}

vec3 vec3::operator-() const {
    return vec3(-e[0], -e[1], -e[2]);
}

datatype vec3::operator[](int i) const {
    return e[i];
}

datatype& vec3::operator[](int i) {
    return e[i];
}

vec3& vec3::operator+=(const vec3& rhs) {
    e[0] += rhs.e[0];
    e[1] += rhs.e[1];
    e[2] += rhs.e[2];
    return *this;
}

vec3& vec3::operator*=(datatype rhs) {
    e[0] *= rhs;
    e[1] *= rhs;
    e[2] *= rhs;
    return *this;
}

vec3& vec3::operator/=(datatype rhs) {
    return *this *= 1 / rhs;
}

datatype vec3::length() const {
    return std::sqrt(length_squared());
}

datatype vec3::length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}