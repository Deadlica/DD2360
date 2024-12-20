// Project
#include <hittable.cuh>

__device__ void hit_record::set_face_normal(const ray& r, const vec3& outward_normal) {
    // "outward_normal" is assumed to have unit length
    front_face = dot(r.direction(), outward_normal) < datatype(0.0);
    normal = front_face ? outward_normal : -outward_normal;
}
