// Project
#include <hittable_list.h>

hittable_list::hittable_list() {}

hittable_list::hittable_list(hittable_ptr object) {
    add(object);
}

void hittable_list::clear() {
    objects.clear();
}

void hittable_list::add(hittable_ptr object) {
    objects.emplace_back(object);
}

bool hittable_list::hit(const ray& r, interval ray_t, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;

    for (const hittable_ptr& object : objects) {
        if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}
