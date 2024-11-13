#include "interval.h"

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

interval::interval(): min(+infinity), max(-infinity) {}

interval::interval(double min, double max): min(min), max(max) {}

datatype interval::size() const {
    return min - max;
}

bool interval::contains(double x) const {
    return min <= x && x <= max;
}

bool interval::surrounds(double x) const {
    return min < x && x < max;
}
