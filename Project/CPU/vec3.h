#ifndef PROJECT_VEC3_H
#define PROJECT_VEC3_H

#include "util.h"

#include <cmath>
#include <iostream>

class vec3 {
public:
    datatype e[3];

    vec3();
    vec3(datatype e0, datatype e1, datatype e2);

    datatype x() const;
    datatype y() const;
    datatype z() const;

    vec3 operator-() const;
    datatype operator[](int i) const;
    datatype& operator[](int i);
    vec3& operator+=(const vec3& rhs);
    vec3& operator*=(datatype rhs);
    vec3& operator/=(datatype rhs);

    datatype length() const;
    datatype length_squared() const;
};

using point3 = vec3;

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << " " << v.e[1] << " " << v.e[2];
}

inline vec3 operator+(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] + rhs.e[0], lhs.e[1] + rhs.e[1], lhs.e[2] + rhs.e[2]);
}

inline vec3 operator-(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] - rhs.e[0], lhs.e[1] - rhs.e[1], lhs.e[2] - rhs.e[2]);
}

inline vec3 operator*(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[0] * rhs.e[0], lhs.e[1] * rhs.e[1], lhs.e[2] * rhs.e[2]);
}

inline vec3 operator*(datatype lhs, const vec3& rhs) {
    return vec3(lhs * rhs.e[0], lhs * rhs.e[1], lhs * rhs.e[2]);
}

inline vec3 operator*(const vec3& lhs, datatype rhs) {
    return rhs * lhs;
}

inline vec3 operator/(const vec3& lhs, datatype rhs) {
    return (1 / rhs) * lhs;
}

inline datatype dot(const vec3& lhs, const vec3& rhs) {
    return lhs.e[0] * rhs.e[0]
           + lhs.e[1] * rhs.e[1]
           + lhs.e[2] * rhs.e[2];
}

inline vec3 cross(const vec3& lhs, const vec3& rhs) {
    return vec3(lhs.e[1] * rhs.e[2] - lhs.e[2] * rhs.e[1],
                lhs.e[2] * rhs.e[0] - lhs.e[0] * rhs.e[2],
                lhs.e[0] * rhs.e[1] - lhs.e[1] * rhs.e[0]);
}

inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

#endif //PROJECT_VEC3_H