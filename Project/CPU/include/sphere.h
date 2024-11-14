#ifndef CPU_SPHERE_H
#define CPU_SPHERE_H

// Project
#include <hittable.h>
#include <vec3.h>

class sphere : public hittable {
public:
    sphere(const point3& center, datatype radius, material_ptr mat);

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

private:
    point3 _center;
    datatype _radius;
    material_ptr _mat;
};


#endif //CPU_SPHERE_H
