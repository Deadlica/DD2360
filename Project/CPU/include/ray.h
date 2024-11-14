#ifndef PROJECT_RAY_H
#define PROJECT_RAY_H

#include "vec3.h"

class ray {
public:
    ray();
    ray(const point3& origin, const vec3& direction);

    const point3& origin() const;
    const vec3& direction() const;

    point3 at(datatype t) const;

private:
    point3 _origin;
    vec3 _direction;
};


#endif //PROJECT_RAY_H
