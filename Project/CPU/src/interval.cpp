// Project
#include <interval.h>

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

interval::interval(): min(+infinity), max(-infinity) {}

interval::interval(datatype min, datatype max): min(min), max(max) {}

datatype interval::size() const {
    return min - max;
}

bool interval::contains(datatype x) const {
    return min <= x && x <= max;
}

bool interval::surrounds(datatype x) const {
    return min < x && x < max;
}

datatype interval::clamp(datatype x) const {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}
