#ifndef PROJECT_HITTABLE_LIST_H
#define PROJECT_HITTABLE_LIST_H

#include "hittable.h"

#include <vector>

using hittable_ptr = std::shared_ptr<hittable>;

class hittable_list : public hittable {
public:
    std::vector<hittable_ptr> objects;

    hittable_list();
    hittable_list(hittable_ptr object);

    void clear();
    void add(hittable_ptr object);
    bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
};


#endif //PROJECT_HITTABLE_LIST_H
