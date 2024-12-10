#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

static constexpr unsigned int TPB = 256;

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}

/** allocate particles, grids and fields on GPU */
void allocate_gpu(struct particles* part, struct EMfield* field, struct grid* grd,
                  size_t particle_size, size_t grid_size,
                  FPpart*& dx,    FPpart*& dy,    FPpart*& dz,
                  FPpart*& du,    FPpart*& dv,    FPpart*& dw,
                  FPfield*& DXN,  FPfield*& DYN,  FPfield*& DZN,
                  FPfield*& DEx,  FPfield*& DEy,  FPfield*& DEz,
                  FPfield*& DBxn, FPfield*& DByn, FPfield*& DBzn)
{
    checkCudaErrors(cudaMalloc(&dx, particle_size));
    checkCudaErrors(cudaMalloc(&dy, particle_size));
    checkCudaErrors(cudaMalloc(&dz, particle_size));

    checkCudaErrors(cudaMalloc(&du, particle_size));
    checkCudaErrors(cudaMalloc(&dv, particle_size));
    checkCudaErrors(cudaMalloc(&dw, particle_size));

    checkCudaErrors(cudaMalloc(&DXN, grid_size));
    checkCudaErrors(cudaMalloc(&DYN, grid_size));
    checkCudaErrors(cudaMalloc(&DZN, grid_size));

    checkCudaErrors(cudaMalloc(&DEx, grid_size));
    checkCudaErrors(cudaMalloc(&DEy, grid_size));
    checkCudaErrors(cudaMalloc(&DEz, grid_size));

    checkCudaErrors(cudaMalloc(&DBxn, grid_size));
    checkCudaErrors(cudaMalloc(&DByn, grid_size));
    checkCudaErrors(cudaMalloc(&DBzn, grid_size));

    checkCudaErrors(cudaMemcpy(dx, part->x, particle_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dy, part->y, particle_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dz, part->z, particle_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(du, part->u, particle_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dv, part->v, particle_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dw, part->w, particle_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(DXN, grd->XN_flat, grid_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(DYN, grd->YN_flat, grid_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(DZN, grd->ZN_flat, grid_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(DEx, field->Ex_flat, grid_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(DEy, field->Ey_flat, grid_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(DEz, field->Ez_flat, grid_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(DBxn, field->Bxn_flat, grid_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(DByn, field->Byn_flat, grid_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(DBzn, field->Bzn_flat, grid_size, cudaMemcpyHostToDevice));
}

/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** deallocate on GPU */
void deallocate_gpu(FPpart* dx,  FPpart* dy,  FPpart* dz,
                    FPpart* du,  FPpart* dv,  FPpart* dw,
                    FPfield* DXN, FPfield* DYN, FPfield* DZN,
                    FPfield* DEx, FPfield* DEy, FPfield* DEz,
                    FPfield* DBxn, FPfield* DByn, FPfield* DBzn)
{
    checkCudaErrors(cudaFree(dx));
    checkCudaErrors(cudaFree(dy));
    checkCudaErrors(cudaFree(dz));

    checkCudaErrors(cudaFree(du));
    checkCudaErrors(cudaFree(dv));
    checkCudaErrors(cudaFree(dw));

    checkCudaErrors(cudaFree(DXN));
    checkCudaErrors(cudaFree(DYN));
    checkCudaErrors(cudaFree(DZN));

    checkCudaErrors(cudaFree(DEx));
    checkCudaErrors(cudaFree(DEy));
    checkCudaErrors(cudaFree(DEz));

    checkCudaErrors(cudaFree(DBxn));
    checkCudaErrors(cudaFree(DByn));
    checkCudaErrors(cudaFree(DBzn));
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}

__global__ void mover_PC_kernel(long     nop,       double   dt,        double   c,
                                int      nyn,       int      nzn,       int      n_sub_cycles,
                                FPpart   qom,       FPfield  invVOL,    int      NiterMover,
                                double   xStart,    double   yStart,    double   zStart,
                                FPfield  invdx,     FPfield  invdy,     FPfield  invdz,
                                double   Lx,        double   Ly,        double   Lz,
                                bool     PERIODICX, bool     PERIODICY, bool     PERIODICZ,
                                FPpart*  x,         FPpart*  y,         FPpart*  z,
                                FPpart*  u,         FPpart*  v,         FPpart*  w,
                                FPfield* XN,        FPfield* YN,        FPfield* ZN,
                                FPfield* Ex,        FPfield* Ey,        FPfield* Ez,
                                FPfield* Bxn,       FPfield* Byn,       FPfield* Bzn)
{
    FPpart dt_sub_cycling = (FPpart) dt / ((double) n_sub_cycles);
    FPpart dto2 = 0.5 * dt_sub_cycling;
    FPpart qomdt2 = qom * dto2 / c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nop) return;

    xptilde = x[i];
    yptilde = y[i];
    zptilde = z[i];
    // calculate the average velocity iteratively
    for(int innter=0; innter < NiterMover; innter++){
        // interpolation G-->P
        ix = 2 +  int((x[i] - xStart)*invdx);
        iy = 2 +  int((y[i] - yStart)*invdy);
        iz = 2 +  int((z[i] - zStart)*invdz);

        // calculate weights
        xi[0]   = x[i] - XN[get_idx(ix - 1, iy, iz, nyn, nzn)];
        eta[0]  = y[i] - YN[get_idx(ix, iy - 1, iz, nyn, nzn)];
        zeta[0] = z[i] - ZN[get_idx(ix, iy, iz - 1, nyn, nzn)];
        xi[1]   = XN[get_idx(ix, iy, iz, nyn, nzn)] - x[i];
        eta[1]  = YN[get_idx(ix, iy, iz, nyn, nzn)] - y[i];
        zeta[1] = ZN[get_idx(ix, iy, iz, nyn, nzn)] - z[i];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * invVOL;

        // set to zero local electric and magnetic field
        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    Exl += weight[ii][jj][kk]*Ex[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)];
                    Eyl += weight[ii][jj][kk]*Ey[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)];
                    Ezl += weight[ii][jj][kk]*Ez[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)];
                    Bxl += weight[ii][jj][kk]*Bxn[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)];
                    Byl += weight[ii][jj][kk]*Byn[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)];
                    Bzl += weight[ii][jj][kk]*Bzn[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)];
                }

        // end interpolation
        omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);
        // solve the position equation
        ut= u[i] + qomdt2*Exl;
        vt= v[i] + qomdt2*Eyl;
        wt= w[i] + qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;
        // solve the velocity equation
        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
        // update position
        x[i] = xptilde + uptilde*dto2;
        y[i] = yptilde + vptilde*dto2;
        z[i] = zptilde + wptilde*dto2;


    } // end of iteration
    // update the final position and velocity
    u[i]= 2.0*uptilde - u[i];
    v[i]= 2.0*vptilde - v[i];
    w[i]= 2.0*wptilde - w[i];
    x[i] = xptilde + uptilde*dt_sub_cycling;
    y[i] = yptilde + vptilde*dt_sub_cycling;
    z[i] = zptilde + wptilde*dt_sub_cycling;


    //////////
    //////////
    ////////// BC

    // X-DIRECTION: BC particles
    if (x[i] > Lx){
        if (PERIODICX==true){ // PERIODIC
            x[i] = x[i] - Lx;
        } else { // REFLECTING BC
            u[i] = -u[i];
            x[i] = 2*Lx - x[i];
        }
    }

    if (x[i] < 0){
        if (PERIODICX==true){ // PERIODIC
            x[i] = x[i] + Lx;
        } else { // REFLECTING BC
            u[i] = -u[i];
            x[i] = -x[i];
        }
    }


    // Y-DIRECTION: BC particles
    if (y[i] > Ly){
        if (PERIODICY==true){ // PERIODIC
            y[i] = y[i] - Ly;
        } else { // REFLECTING BC
            v[i] = -v[i];
            y[i] = 2*Ly - y[i];
        }
    }

    if (y[i] < 0){
        if (PERIODICY==true){ // PERIODIC
            y[i] = y[i] + Ly;
        } else { // REFLECTING BC
            v[i] = -v[i];
            y[i] = -y[i];
        }
    }

    // Z-DIRECTION: BC particles
    if (z[i] > Lz){
        if (PERIODICZ==true){ // PERIODIC
            z[i] = z[i] - Lz;
        } else { // REFLECTING BC
            w[i] = -w[i];
            z[i] = 2*Lz - z[i];
        }
    }

    if (z[i] < 0){
        if (PERIODICZ==true){ // PERIODIC
            z[i] = z[i] + Lz;
        } else { // REFLECTING BC
            w[i] = -w[i];
            z[i] = -z[i];
        }
    }
}

int mover_PC_gpu(struct particles* part, struct EMfield* field,
                 struct grid* grd, struct parameters* param)
{
    // Allocate to GPU
    size_t particle_size = sizeof(FPpart)  * part->npmax;
    size_t grid_size     = sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn;

    FPpart*  dx;   FPpart*  dy;   FPpart*  dz;
    FPpart*  du;   FPpart*  dv;   FPpart*  dw;
    FPfield* DXN;  FPfield* DYN;  FPfield* DZN;
    FPfield* DEx;  FPfield* DEy;  FPfield* DEz;
    FPfield* DBxn; FPfield* DByn; FPfield* DBzn;

    allocate_gpu(part, field, grd,
                 particle_size, grid_size,
                 dx, dy, dz,
                 du, dv, dw,
                 DXN, DYN, DZN,
                 DEx, DEy, DEz,
                 DBxn, DByn, DBzn);

    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    dim3 db(TPB);
    dim3 dg((part->nop + db.x - 1) / db.x);

    for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
        mover_PC_kernel<<<dg, db>>>(part->nop,      param->dt,      param->c,
                                    grd->nyn,       grd->nzn,       param->n_sub_cycles,
                                    part->qom,      grd->invVOL,    part->NiterMover,
                                    grd->xStart,    grd->yStart,    grd->zStart,
                                    grd->invdx,     grd->invdy,     grd->invdz,
                                    grd->Lx,        grd->Ly,        grd->Lz,
                                    grd->PERIODICX, grd->PERIODICY, grd->PERIODICZ,
                                    dx,             dy,             dz,
                                    du,             dv,             dw,
                                    DXN,            DYN,            DZN,
                                    DEx,            DEy,            DEz,
                                    DBxn,           DByn,           DBzn);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Copy results back to CPU
    checkCudaErrors(cudaMemcpy(part->x, dx, particle_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(part->y, dy, particle_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(part->z, dz, particle_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(part->u, du, particle_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(part->v, dv, particle_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(part->w, dw, particle_size, cudaMemcpyDeviceToHost));

    // Free GPU
    deallocate_gpu(dx, dy, dz,
                   du, dv, dw,
                   DXN, DYN, DZN,
                   DEx, DEy, DEz,
                   DBxn, DByn, DBzn);

    return (0);
}