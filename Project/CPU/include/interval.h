#ifndef CPU_INTERVAL_H
#define CPU_INTERVAL_H

// Project
#include <util.h>
#include <vec3.h>

class interval {
public:
    datatype min, max;

    interval();
    interval(datatype min, datatype max);

    datatype size() const;
    bool contains(datatype x) const;
    bool surrounds(datatype x) const;
    datatype clamp(datatype x) const;

    static const interval empty, universe;
};


#endif //CPU_INTERVAL_H
