#ifndef PROJECT_MATERIAL_H
#define PROJECT_MATERIAL_H

#include "hittable.h"
#include "color.h"

class material {
public:
    virtual ~material() = default;

    virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const;
};

class lambertian : public material {
public:
    lambertian(const color& albedo);

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override;

private:
    color _albedo;
};

class metal : public material {
public:
    metal(const color& albedo, datatype fuzz);

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override;

private:
    color _albedo;
    datatype _fuzz;
};

#endif //PROJECT_MATERIAL_H
