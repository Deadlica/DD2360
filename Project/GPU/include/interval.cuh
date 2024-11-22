#ifndef GPU_INTERVAL_CUH
#define GPU_INTERVAL_CUH

// Project
#include <util.cuh>
#include <vec3.cuh>

/**
 * @class interval
 * @brief Represents a numerical range with a minimum and maximum value.
 *
 * The `interval` class is used to represent a range of values, typically used for ray tracing to track
 * the potential intersection times or other numerical ranges. It provides methods for checking containment
 * and clamping values within the range.
 */
class interval {
public:
    datatype min, max;

    /**
    * @brief Default constructor, initializes the interval to an empty range.
    */
    __host__ __device__ interval();

    /**
     * @brief Constructs an interval with a specified minimum and maximum value.
     *
     * @param min The minimum value of the interval.
     * @param max The maximum value of the interval.
     */
    __host__ __device__ interval(datatype min, datatype max);

    /**
     * @brief Gets the size of the interval.
     *
     * @return The size of the interval (max - min).
     */
    __device__ datatype size() const;

    /**
     * @brief Checks if a value is contained within the interval.
     *
     * @param x The value to check.
     * @return `true` if the value is within the interval, otherwise `false`.
     */
    __device__ bool contains(datatype x) const;

    /**
     * @brief Checks if the interval surrounds a value.
     *
     * @param x The value to check.
     * @return `true` if the value is outside the interval but the interval surrounds it, otherwise `false`.
     */
    __device__ bool surrounds(datatype x) const;

    /**
     * @brief Clamps a value within the bounds of the interval.
     *
     * @param x The value to clamp.
     * @return The clamped value within the interval.
     */
    __host__ __device__ datatype clamp(datatype x) const;

    static const interval empty, universe;
};

#endif //GPU_INTERVAL_CUH
