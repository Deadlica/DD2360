#ifndef GPU_PRECISION_CUH
#define GPU_PRECISION_CUH

#include <cfloat>
#include <cuda_fp16.h>

#define datatype float

#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif

#if   defined(datatype) && datatype == float
    #define infinity FLT_MAX
#elif defined(datatype) && datatype == double
    #define infinity DBL_MAX
#elif defined(datatype) && datatype == __half
    #define infinity 65504.0f
#endif

#endif //GPU_PRECISION_CUH
