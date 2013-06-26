"""
Grady Williams
June 19, 2013

GPU Implementation of Multi-PI^2 for simulating quad-rotor navigation of a
forest
"""

import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import *
from pycuda import gpuarray
import math
from datetime import datetime
import pickle


"""
Define some parameters regarding the world
"""
ceiling_height = 10
floor = 0
control_dimensions = 4
state_dimensions = 16
world_radius = 10
sub_forest_radius = 10


"""
Define some parameters for the virtual quad-rotor
"""
M = .15
m = .02
total_mass = .25
l = .1
R = .05
I_x = .4*(M*R**2 + l**2*m)
I_y = I_x
I_z = .4*M*R**2 + 4*l**2*m
I = np.array([[I_x,0,0],[0,I_y,0],[0,0,I_z]])
K_m = 1.5e-9
K_f = 6.11e-8
motor_gain = 1.0

"""
PI^2 class for performing the PI^2 algorithm in the context of quadrotor navigation
on a CUDA capable GPU
"""
class M_Pi_2:

    """
    Initializes the PI^2 class, compiles all the CUDA code and stores them
    as callable functions
    """
    def __init__(self, init_pos, forest, K, time, T):
        self.init_state = np.require(init_pos, dtype=np.float32, requirements = ['A', 'O', 'W', 'C'])
        self.init_state_d = gpuarray.to_gpu(init_state)
        self.K = K
        self.time = time
        self.T = T
        self.dt = time/(1.0*T)
        self.generator = XORWOWRandomNumberGenerator()
        self.controls_d = gpuarray.zeros(T*K*control_dimensions, dtype = np.float32)
        self.state_costs_d = gpuarray.zeros(T*K, dtype = np.float32)
        self.terminal_costs_d = gpuarray.zeros(K, dtype = np.float32)
        self.forest = self.make_forest(forest)
        #Initialize CUDA functions
        rollout_kernel, U_d = self.func1(T)
        cost_to_go = self.func2(T)
        reduction = self.func3()
        multiply = self.func4()
        self.funcs = rollout_kernel, cost_to_go, reduction, multiply, U_d

    def rollouts(self, U, var1, var2, test = False):
        #Get an array of random numbers from CUDA. The numbers are
        #normally distributed with mean = 0 and variance = var
        du = self.generator.gen_normal(self.K*self.T*(control_dimensions), np.float32)
        #Get Tree Information!!
        trees = gpuarray.zeros(10, dtype=np.float32)
        #Unpack CUDA functions from self.funcs, get addresses of array in constant 
        #memory
        rollout_kernel, cost_to_go, reduction, multiply, U_d = self.funcs
        cuda.memcpy_dtod(U_d, U.ptr, U.nbytes)
        #Set blocksize and gridsize for rollout and cost-to-go kernels
        blocksize = (64, 1, 1)
        gridsize = ((self.K - 1)//64 + 1, 1, 1)
        #print "Rollout Kernel"
        start = datetime.now()
        #Launch the kernel for simulating rollouts
        z_locs = gpuarray.zeros(self.K*self.T, dtype=np.float32)
        w_speed = gpuarray.zeros(self.K*self.T, dtype=np.float32)
        cuda.Context.synchronize()
        rollout_kernel(self.controls_d, self.state_costs_d, self.terminal_costs_d,
                       du, self.init_state_d, trees, np.float32(self.dt), 
                       np.float32(motor_gain),
                       np.float32(total_mass), np.float32(I_x), np.float32(I_y),
                       np.float32(I_z), np.float32(l), np.float32(var1), np.float32(var2),
                       z_locs, w_speed,
                       grid=gridsize, block=blocksize)
        cuda.Context.synchronize()
        """
        print "State Costs"
        print self.state_costs_d
        print "Terminal Costs"
        print self.terminal_costs_d
        print "Controls"
        print self.controls_d
        print "Phi Angle"
        print z_locs
        print
        print
        """
        d = self.terminal_costs_d.get()
        d = np.mean(d)
        #Launch the kernel for computing cost-to-go values for each state
        blocksize = (self.T,1,1)
        gridsize = (self.K,1,1)
        cost_to_go(self.state_costs_d, self.terminal_costs_d, block=blocksize, grid=gridsize)
        cuda.Context.synchronize()
        #Compute the normalizer, the normalizer is an array with T indices which
        #contains the sums of columns of state_costs_d        
        T_int = int(self.T)
        K_int = int(self.K)
        start = datetime.now()
        j = (K_int-1)//16 + 1
        out_d = gpuarray.zeros(T_int*j, np.float32)
        gridsize = ((T_int-1)//16 + 1, j, 1)
        blocksize = (16, 16, 1) 
        reduction(self.state_costs_d, out_d, np.int32(K_int), np.int32(T_int), grid=gridsize, block=blocksize)
        cuda.Context.synchronize()
        while (j > 1):
            _k = j
            j = (j-1)//16 + 1
            in_d = out_d
            out_d = gpuarray.zeros(T_int*j, np.float32)
            gridsize = ((T_int-1)//16 + 1, _k, 1)
            reduction(in_d, out_d, np.int32(_k), np.int32(T_int), grid=gridsize, block=blocksize)
            cuda.Context.synchronize()
        normalizer = out_d
        cuda.Context.synchronize()
        #Multipy the controls by the weighted score. The weighted score is the cost-to-go
        #function divided by the normalizer.
        start = datetime.now()
        blocksize = (16,16,1)
        gridsize = ((T_int-1)//16+1, (K_int-1)//16 + 1,1)
        multiply(normalizer, self.controls_d, self.state_costs_d, np.int32(T_int), np.int32(K_int), 
                block=blocksize, grid=gridsize)
        cuda.Context.synchronize()
        start = datetime.now()
        #Launch the sum reduction code to compute the weighted average of all the paths
        j = (K_int-1)//16 + 1
        out_d = gpuarray.zeros(T_int*control_dimensions*j, np.float32)
        gridsize = ((T_int*control_dimensions-1)//16 + 1, j, 1)
        blocksize = (16, 16, 1) 
        reduction(self.controls_d, out_d, np.int32(K_int), np.int32(T_int*control_dimensions), 
                 grid=gridsize, block=blocksize)
        cuda.Context.synchronize()
        while (j > 1):
            _k = j
            j = (j-1)//16 + 1
            in_d = out_d
            out_d = gpuarray.zeros(control_dimensions*T_int*j, np.float32)
            gridsize = ((control_dimensions*T_int - 1)//16 + 1, _k, 1)
            reduction(in_d, out_d, np.int32(_k), np.int32(control_dimensions*T_int), 
                     grid=gridsize, block=blocksize)
            cuda.Context.synchronize()
        #print (datetime.now() - start).microseconds/1000.0
        return out_d,d
    
    """
    Answers the a function call
    """
    def calc_path(self):
        return 0

    def plotter(self, U, init_state):
        dt = self.dt
        T = self.T
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.zeros(T)
        Y = np.zeros(T)
        Z = np.zeros(T)
        s = init_state
        hover_speed = 3167.77
        w = np.zeros(4)
        max_speed = 9000
        min_speed = 1000
        for t in range(T):
            w[0] = hover_speed + U[t,0] - U[t,2] + U[t,3];
            w[1] = hover_speed + U[t,0] + U[t,1] - U[t,3];
            w[2] = hover_speed + U[t,0] + U[t,2] + U[t,3];
            w[3] = hover_speed + U[t,0] - U[t,1] + U[t,3];
            if (w[0] > max_speed):
                w[0] = max_speed
            elif (w[0] < min_speed):
                w[0] = min_speed
            if (w[1] > max_speed):
                w[1] = max_speed
            elif (w[1] < min_speed):
                w[1] = min_speed
            if (w[2] > max_speed):
                w[2] = max_speed
            elif (w[2] < min_speed):
                w[2] = min_speed
            if (w[3] > max_speed):
                w[3] = max_speed
            elif (w[3] < min_speed):
                w[3] = min_speed
           
            print 
            print "----------------------------------------"
            print
            print w
            if (abs(s[3]) > 1 or abs(s[4]) > 1):
                print "DANGER"
            s[0] += dt*s[6]
            s[1] += dt*s[7]
            s[2] += dt*s[8]
            X[t] = s[0]
            Y[t] = s[1]
            Z[t] = s[2]
            if (s[2] < 0):
                s[2] = 0
            s[3] += dt*(np.cos(s[4])*s[9] + np.sin(s[4])*s[11])
            s[4] += dt*(np.sin(s[4])*np.tan(s[3])*s[9] + s[10] - np.cos(s[4])*np.tan(s[3])*s[11])
            s[5] += dt*(-np.sin(s[4])/np.cos(s[3])*s[9] + np.cos(s[4])/np.cos(s[3])*s[11])
            F_1 = K_f * s[12]**2
            F_2 = K_f * s[13]**2
            F_3 = K_f * s[14]**2
            F_4 = K_f * s[15]**2
            F_sum = (F_1 + F_2 + F_3 + F_4)/total_mass
            s[6] += dt*F_sum*(np.cos(s[5])*np.sin(s[4]) + np.cos(s[4])*np.sin(s[5])*np.sin(s[3]))
            s[7] += dt*F_sum*(np.sin(s[5])*np.sin(s[4]) - np.cos(s[5])*np.cos(s[4])*np.sin(s[3]))
            s[8] += dt*(F_sum*(np.cos(s[3])*np.cos(s[4])) - 9.81)
            s[9] += dt/I_x * (l*(F_2 - F_4) - s[10]*s[11]*(I_z - I_y))
            s[10] += dt/I_y * (l*(F_3 - F_1) - s[9]*s[11]*(I_x - I_z))
            M_1 = K_m*s[12]**2
            M_2 = K_m*s[13]**2
            M_3 = K_m*s[14]**2
            M_4 = K_m*s[15]**2
            s[11] += dt/I_z * (M_1 - M_2 + M_3 - M_4 - s[9]*s[10]*(I_y - I_x))
            s[12] += motor_gain*dt*(w[0] - s[12])
            s[13] += motor_gain*dt*(w[1] - s[13])
            s[14] += motor_gain*dt*(w[2] - s[14])
            s[15] += motor_gain*dt*(w[3] - s[15])
            #print s
        ax.scatter(X,Y,Z)
        plt.show()
            
    
    def make_forest(self, forest):
        return 0
    
    def get_sub_forest(self, forest, pos):
        return 0

    
    """==========================================================================================="""
    #################################################################################################
    """==========================================================================================="""
    
    """
    CUDA functions:
    
    func1 = Function for computing rollouts
    func2 = Function for computing cost-to-go functions 
    func3 = Sum reduction function (Used for computing the normalizing term 
            and for computing the new control vector.
    func4 = Function for multiplying states by their weights
    """
    
    def func1(self, T):
        template = """
   
    #include <math.h>
    #include <stdio.h>
    
    #define T %d
    #define CEILING_H %d
    #define WORLD_R %d
    #define CONTROL_DIM %d
    #define STATE_DIM %d
    #define K %d
    
    __device__ __constant__ float U_d[T*CONTROL_DIM];
    
    __device__ int get_distance(float* state) 
    {
       return 0;
    }

    __device__ float get_state_cost(float* s, float* forces)
    { 
       float ground_penalty = 0;
       if (s[2]*s[2] < .1) {
          ground_penalty = 4;
       }
       float tilt_penalty = 0;
       if (s[3] > 1) {
         tilt_penalty += 50;
       }
       if (s[4] > 1) {
          tilt_penalty += 50;
       }
       return ground_penalty + (s[2] - 1)*(s[2] - 1) + (s[0] - 1)*(s[0]-1) + (s[1] - 1)*(s[1]-1);
    }

    __device__ float get_terminal_cost(float* s)
    {
       return 500*( (s[2] - 1)*(s[2] - 1) + (s[0] - 1)*(s[0]-1) + (s[1] - 1)*(s[1]-1) );
    }
   
    
    /*******************************************
    
    Kernel Function for Computing Rollouts
  
    *******************************************/
   
    __global__ void rollout_kernel(float* controls_d, float* state_costs_d, float* terminal_costs_d,
                                   float* du, float* init_state, float* trees, float dt, float motor_gain, 
                                   float mass, float Ix, float Iy, float Iz, float l, 
                                   float var1, float var2, float* z_locs, float* w_speed)
    {
       //Get thread and block index
       int tdx = threadIdx.x;
       int bdx = blockIdx.x;
       float gravity = 9.81;
       float max_speed = 9000;
       float min_speed = 1000;
       //Initialize local state
       float s[STATE_DIM];
       float u[CONTROL_DIM];
       float w[CONTROL_DIM];
       float hover_speed = 3167.77;
       float km = .00015;
       float kf = .00611;
       int i,j;
       for (i = 0; i < STATE_DIM; i++) {
          s[i] = init_state[i];
       }
       if (64*bdx + tdx < K) {
          for (i = 0; i < T; i++) {
             for (j = 0; j < CONTROL_DIM; j++) {
                if (j %% 4 == 0) {
                   u[j] = U_d[i*CONTROL_DIM + j] + du[(64*bdx + tdx)*T*CONTROL_DIM + i*CONTROL_DIM + j]*var1;
                }
                else {
                    u[j] = U_d[i*CONTROL_DIM + j] + du[(64*bdx + tdx)*T*CONTROL_DIM + i*CONTROL_DIM + j]*var2;
                }
             }
             w[0] = hover_speed + u[0] - u[2] + u[3];
             w[1] = hover_speed + u[0] + u[1] - u[3];
             w[2] = hover_speed + u[0] + u[2] + u[3];
             w[3] = hover_speed + u[0] - u[1] + u[3];
             if (w[0] > max_speed) {
                w[0] = max_speed;
             }
             else if (w[0] < min_speed) {
                w[0] = min_speed;
             }
             if (w[1] > max_speed) {
                w[1] = max_speed;
             }
             else if (w[1] < min_speed) {
                w[1] = min_speed;
             }
             if (w[2] > max_speed) {
                w[2] = max_speed;
             }
             else if (w[2] < min_speed) {
                w[2] = min_speed;
             }
             if (w[3] > max_speed) {
                w[3] = max_speed;
             }
             else if (w[3] < min_speed) {
                w[3] = min_speed;
             }
             s[0] += dt*s[6];
             s[1] += dt*s[7];
             s[2] += dt*s[8];
             if (s[2] < 0){
                 s[2] = 0;
             }
             //if (s[2] > CEILING_H) {
             //   s[2] = CEILING_H;
             //}
             s[3] += dt*(cos(s[4])*s[9] + sin(s[4])*s[11]);
             s[4] += dt*(sin(s[4])*tan(s[3])*s[9] + s[10] - cos(s[4])*tan(s[3])*s[11]);
             s[5] += dt*(-sin(s[4])/cos(s[3])*s[9] + cos(s[4])/cos(s[3])*s[11]);
             float F_1 = kf * s[12]*(.00001)*s[12];
             float F_2 = kf * s[13]*(.00001)*s[13];
             float F_3 = kf * s[14]*(.00001)*s[14];
             float F_4 = kf * s[15]*(.00001)*s[15];
             float F_sum = (F_1 + F_2 + F_3 + F_4)/mass;
             s[6] += dt*F_sum*(cos(s[5])*sin(s[4]) + cos(s[4])*sin(s[5])*sin(s[3]));
             s[7] += dt*F_sum*(sin(s[5])*sin(s[4]) - cos(s[5])*cos(s[4])*sin(s[3]));
             s[8] += dt*(F_sum*(cos(s[3])*cos(s[4])) - gravity);
             s[9] += dt/Ix * (l*(F_2 - F_4) - s[10]*s[11]*(Iz - Iy));
             s[10] += dt/Iy * (l*(F_3 - F_1) - s[9]*s[11]*(Ix - Iz));
             float M_1 = km*s[12]*(.00001)*s[12];
             float M_2 = km*s[13]*(.00001)*s[13];
             float M_3 = km*s[14]*(.00001)*s[14];
             float M_4 = km*s[15]*(.00001)*s[15];
             s[11] += dt/Iz * (M_1 - M_2 + M_3 - M_4 - s[9]*s[10]*(Iy - Ix));
             s[12] += dt*motor_gain*(w[0] - s[12]);
             s[13] += dt*motor_gain*(w[1] - s[13]);
             s[14] += dt*motor_gain*(w[2] - s[14]);
             s[15] += dt*motor_gain*(w[3] - s[15]);
             //Get State Costs
             float cost = get_state_cost(s, u);
             //Add costs into state_costs
             state_costs_d[(64*bdx + tdx)*T + i] = cost;
             w_speed[(64*bdx + tdx)*T + i] = s[12];
             z_locs[(64*bdx + tdx)*T + i] = s[3];
             //Add controls
             for (j = 0; j < CONTROL_DIM; j++) {
                controls_d[(64*bdx + tdx)*T*CONTROL_DIM + i*CONTROL_DIM + j] = u[j];
             }
          }
          //Get Terminal Costs
          float terminal_cost = get_terminal_cost(s);
          terminal_costs_d[64*bdx + tdx] = terminal_cost;
       }
    }
    """%(T, ceiling_height, world_radius, control_dimensions, state_dimensions, self.K)
        mod = SourceModule(template)
        func = mod.get_function("rollout_kernel")
        U_d = mod.get_global("U_d")[0]
        return func, U_d

    
    def func2(self, T):
       template = """
 
    #include <math.h>

    #define T %d

    __global__ void cost_to_go(float* states, float* terminals) 
    {
      int tdx = threadIdx.x;
      int bdx = blockIdx.x;
      __shared__ float s_costs[T];
      s_costs[tdx] = states[bdx*T + tdx];
      __syncthreads();

      if (tdx == 0) {  
        float sum = terminals[bdx];
        int i;
        for (i = 0; i < T; i++) {
          sum += s_costs[T-1-i];
          s_costs[T-1-i] = sum;
        }
      }
      __syncthreads();
      float lambda = -1.0/100.0;
      states[bdx*T+tdx] = exp(lambda*s_costs[tdx]);
    }

    """%T
       mod = SourceModule(template)
       func = mod.get_function("cost_to_go")
       return func
 
    
    def func3(self):
        template = """
    #include <stdio.h>
    
    __global__ void reduction_kernel(float* in_d, float* out_d, int y_len, int x_len)
    {
      int tdx = threadIdx.x;
      int tdy = threadIdx.y;
      int bdx = blockIdx.x;
      int bdy = blockIdx.y;

      int x_ind = bdx*16 + tdx;
      int y_ind = bdy*16 + tdy;

      __shared__ double data[16*16];
      data[16*tdy + tdx] = 0;

      if (x_ind < x_len && y_ind < y_len) {
        data[tdy*16 + tdx] = in_d[y_ind*x_len + x_ind];
      }
      __syncthreads();

      for (int i = 8; i > 0; i>>=1) {
         if (tdy < i){
            data[16*tdy + tdx] += data[16*(tdy+i) + tdx];
         }
        __syncthreads();
      }

      if (tdy == 0 && x_ind < x_len) {
        out_d[bdy*x_len + x_ind] = data[tdx];
      } 
    }
    """
        mod = SourceModule(template)
        return mod.get_function("reduction_kernel")


    """
    Averages all of the paths together
    """
    def func4(self):
        template = """
    #define CONTROL_DIM %d 

    __global__ void multiplier(float *normalizer, float* controls, float* states_d, int x_len, int y_len)
    {
      int tdx = threadIdx.x;
      int tdy = threadIdx.y;
      int bdx = blockIdx.x;
      int bdy = blockIdx.y;

      int x_ind = 16*bdx + tdx;
      int y_ind = 16*bdy + tdy;
      if (x_ind < x_len && y_ind < y_len) {
        float normal = normalizer[x_ind];
        int i;
        float weight = states_d[y_ind * x_len + x_ind]/ normal;
        for (i = 0; i < CONTROL_DIM; i++) {
          controls[(y_ind * x_len + x_ind)*CONTROL_DIM + i] *= weight;
        }
      }
    }
    """%(control_dimensions)
        mod = SourceModule(template)
        return mod.get_function("multiplier")


if __name__ == "__main__":
    init_state = np.zeros(16, dtype=np.float32)
    init_state[12:] = 3167.77
    init_state[2] = 0
    T = 100
    K = 1000
    U = np.zeros(T*4, dtype = np.float32)
    U = np.require(U, dtype=np.float32, requirements = ['A', 'O', 'W', 'C'])
    U = gpuarray.to_gpu(U)
    var1 = 550
    var2 = 1.5
    var11 = var1
    var22 = var2
    forest = [0]
    A = M_Pi_2(init_state, forest, K, 2, T)
    for i in range(30):
        print i   
        U,d = A.rollouts(U,var11, var22)
        print d
        var11 = var1*(d/300.0)
        var22 = var2*(d/300.0)
    #for i in range(T*4):
    #    if (i % 4 == 0):
    #        U[i] = 0
    #    if (i % 4 == 1):
    #        U[i] = 5*(i/(25.0) - 1.0)
    A.plotter(U.get().reshape(T,4), init_state)
