#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"

struct particles {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;
    
    
    
};

/** Macro to check CUDA error and terminate the program on failure */
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
/** Checks the result of a CUDA API call and prints an error message if the call failed */
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

/** allocate particle arrays */
void particle_allocate(struct parameters*, struct particles*, int);

/** allocate particles, grids and fields on GPU */
void allocate_gpu(struct particles*, struct EMfield*, struct grid*,
                  size_t, size_t,
                  FPpart*&,  FPpart*&,  FPpart*&,
                  FPpart*&,  FPpart*&,  FPpart*&,
                  FPfield*&, FPfield*&, FPfield*&,
                  FPfield*&, FPfield*&, FPfield*&,
                  FPfield*&, FPfield*&, FPfield*&);

/** deallocate */
void particle_deallocate(struct particles*);

/** deallocate on GPU */
void deallocate_gpu(FPpart*,  FPpart*,  FPpart*,
                    FPpart*,  FPpart*,  FPpart*,
                    FPfield*, FPfield*, FPfield*,
                    FPfield*, FPfield*, FPfield*,
                    FPfield*, FPfield*, FPfield*);

/** particle mover */
int mover_PC(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/** particle mover kernel on GPU */
int mover_PC_gpu(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles*, struct interpDensSpecies*, struct grid*);

/** particle mover kernel */
__global__ void mover_PC_kernel(long,     double,   double,
                                int,      int,   int,
                                FPpart,   FPfield,  int,
                                double,   double,   double,
                                FPfield,  FPfield,  FPfield,
                                double,   double,   double,
                                bool,     bool,     bool,
                                FPpart*,  FPpart*,  FPpart*,
                                FPpart*,  FPpart*,  FPpart*,
                                FPfield*, FPfield*, FPfield*,
                                FPfield*, FPfield*, FPfield*,
                                FPfield*, FPfield*, FPfield*);

#endif
