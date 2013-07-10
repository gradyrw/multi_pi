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
import time

"""
Define some parameters regarding the world
"""
control_dimensions = 4
state_dimensions = 16
gravity = 9.81
"""
Define some parameters for the virtual quad-rotor
"""
M = .15
m = .02
total_mass = .25
total_radius = .15
l = .1
R = .05
Ix = .4*(M*R**2 + l**2*m)
Iy = Ix
Iz = .4*M*R**2 + 4*l**2*m
K_m = 1.5e-9
K_f = 6.11e-8
motor_gain = 1.0
max_rotor_speed = 9000.0
min_rotor_speed = 1000.0
max_thrust = 5832.23
min_thrust = -2167.77
max_tilt = 4000.0
min_tilt = -4000.0
"""
Define some paramaters for CUDA compilation
"""
block_dim = 64

"""
PI^2 class for performing the PI^2 algorithm in the context of quadrotor navigation
on a CUDA capable GPU
"""
class M_Pi_2:

    """
    Initializes the PI^2 class, compiles all the CUDA code and stores them
    as callable functions
    """
    def __init__(self, init_state, forest, K, time, T, mech_params = [1.5e-9, 6.11e-8, 1.0]):
        #Store the initial stae values on both the CPU and GPU
        self.state = init_state
        self.state_d = self.on_gpu(state)
        #K is the number of rollouts to be performed per trial.
        self.K = K
        #T is the number of timesteps to be taken per rollout
        self.T = T
        #dt is the amount of time take from one time-step to another.
        self.dt = time/(1.0*T)
        #Turn the mech params into a numpy and gpu array
        self.mech_params = np.array(mech_params, dtype = np.float32)
        self.mech_params_d = self.on_gpu(self.mech_params)
        #generator is a gpu function which quickly generates large amounts of normally
        #distributed random numbers.
        self.generator = XORWOWRandomNumberGenerator()
        #Initialize controls, state costs, and terminal costs as GPU arrays.
        self.controls_d = gpuarray.zeros(T*K*control_dimensions, dtype = np.float32)
        self.state_costs_d = gpuarray.zeros(T*K, dtype = np.float32)
        self.terminal_costs_d = gpuarray.zeros(K, dtype = np.float32)
        #Initialize the forest as a CPU and GPU array.
        self.forest = forest
        self.forest_d = self.on_gpu(np.array(forest,dtype=np.float32))
        #Compile and store CUDA functions
        rollout_kernel, U_d = self.func1()
        cost_to_go = self.func2(T)
        reduction = self.func3()
        multiply = self.func4()
        self.funcs = rollout_kernel, cost_to_go, reduction, multiply, U_d

    def rollouts(self, U, var1, var2, goal_state_d, A_d, R_d):
        forest = []
        for i in range(len(self.forest)//4):
            if np.sqrt((self.state[0] - self.forest[4*i])**2 + (self.state[1] - self.forest[4*i+1])**2) < 9:
                forest.extend(self.forest[4*i:4*(i+1)])
        num_trees = len(forest)//4
        if (num_trees is not 0):
            self.forest_d = self.on_gpu(np.array(forest,dtype=np.float32))
        print num_trees
        #Get an array of random numbers from CUDA. The numbers are
        #standard normal, the variance is changed later on in the rollout_kernel
        du_d = self.generator.gen_normal(self.K*self.T*(control_dimensions), np.float32)
        #Unpack CUDA functions from self.funcs, get addresses of current control
        #array in constant memory
        rollout_kernel, cost_to_go, reduction, multiply, U_d = self.funcs
        cuda.memcpy_dtod(U_d, U.ptr, U.nbytes)
        #Set blocksize and gridsize for rollout and cost-to-go kernels
        blocksize = (block_dim, 1, 1)
        gridsize = ((self.K - 1)//block_dim + 1, 1, 1)
        #Launch the kernel for simulating rollouts
        rollout_kernel(self.controls_d, self.state_costs_d, self.terminal_costs_d,
                       du_d, self.state_d, self.forest_d, goal_state_d, A_d, R_d, 
                       self.mech_params_d, np.float32(var1), np.float32(var2), np.int32(num_trees),
                       grid=gridsize, block=blocksize)
        cuda.Context.synchronize()
        #d = self.state_costs_d.get()
        #print "Min of state costs"
        #minim =  np.min(d)
        #d = d - minim
        #self.state_costs_d = self.on_gpu(d)
        #d = self.state_costs_d.get()
        #print "Min of state costs"
        #print np.min(d)
        #Launch the kernel for computing cost-to-go values for each state
        blocksize = (self.T,1,1)
        gridsize = (self.K,1,1)
        start = datetime.now()
        cost_to_go(self.state_costs_d, self.terminal_costs_d, block=blocksize, grid=gridsize)
        cuda.Context.synchronize()
        #Compute the normalizer, the normalizer is an array with T indices which
        #contains the sums of columns of state_costs_d        
        T_int = int(self.T)
        K_int = int(self.K)
        j = (K_int-1)//16 + 1
        out_d = gpuarray.empty(T_int*j, np.float32)
        gridsize = ((T_int-1)//16 + 1, j, 1)
        blocksize = (16, 16, 1) 
        reduction(self.state_costs_d, out_d, np.int32(K_int), np.int32(T_int),
                  grid=gridsize, block=blocksize)
        cuda.Context.synchronize()
        while (j > 1):
            _k = j
            j = (j-1)//16 + 1
            in_d = out_d
            out_d = gpuarray.empty(T_int*j, np.float32)
            gridsize = ((T_int-1)//16 + 1, _k, 1)
            reduction(in_d, out_d, np.int32(_k), np.int32(T_int), grid=gridsize, block=blocksize)
            cuda.Context.synchronize()
        normalizer = out_d
        cuda.Context.synchronize()
        #Multipy the controls by the weighted score. The weighted score is the cost-to-go
        #function divided by the normalizer.
        blocksize = (16,16,1)
        gridsize = ((T_int-1)//16+1, (K_int-1)//16 + 1,1)
        start = datetime.now()
        multiply(normalizer, self.controls_d, self.state_costs_d, np.int32(T_int), np.int32(K_int), 
                block=blocksize, grid=gridsize)
        cuda.Context.synchronize()
        #Launch the sum reduction code to compute the weighted average of all the paths
        j = (K_int-1)//16 + 1
        out_d = gpuarray.empty(T_int*control_dimensions*j, np.float32)
        gridsize = ((T_int*control_dimensions-1)//16 + 1, j, 1)
        blocksize = (16, 16, 1) 
        reduction(self.controls_d, out_d, np.int32(K_int), np.int32(T_int*control_dimensions), 
                 grid=gridsize, block=blocksize)
        cuda.Context.synchronize()
        while (j > 1):
            _k = j
            j = (j-1)//16 + 1
            in_d = out_d
            out_d = gpuarray.empty(control_dimensions*T_int*j, np.float32)
            gridsize = ((control_dimensions*T_int - 1)//16 + 1, _k, 1)
            reduction(in_d, out_d, np.int32(_k), np.int32(control_dimensions*T_int), 
                     grid=gridsize, block=blocksize)
            cuda.Context.synchronize()
        return out_d
    
    """
    Answers the a function call
    """
    def multi_pi(self):
        #Defines starting parameters for the system
        dt = self.dt
        T = self.T
        U = np.zeros(T*4, dtype = np.float32)
        U = self.on_gpu(U)
        #Defines the initial state of the quadrotors
        init_pos = np.zeros(16)
        init_pos[12:] = 3167.77
        self.state= init_pos
        self.state_d = self.on_gpu(init_pos)
        #X,Y, and Z hold location data for plotting
        X = []
        Y = []
        Z = []
        #Defines the control cost matrix and state weighting vector for PI^2
        A = np.zeros(16,dtype=np.float32)
        A[:3] = 5
        A[3:12] = 1
        A *= .5
        R = np.identity(4, dtype = np.float32)*(1/1000000.0)
        A_d = self.on_gpu(A)
        R_d = self.on_gpu(R)
        #Prepares the quadrotor for takeoff, p is the point it's aiming at.
        p = np.array([0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
        p_d = self.on_gpu(p)
        U = np.zeros(T*4, dtype = np.float32)
        U = self.on_gpu(U)
        var = 400
        #Takeoff!
        while(abs(2 - self.state[2]) > .05):
            start = time.time()
            U = self.rollouts(U, var, 1.5, p_d, A_d, R_d)
            U_old = U.get()
            print time.time() - start
            info = self.simulator(U_old.reshape(T,4), self.state, 1)
            self.state= info[0]
            self.state_d = self.on_gpu(info[0])
            X.extend(info[1])
            Y.extend(info[2])
            Z.extend(info[3])
            U_new = np.zeros(T*4, dtype = np.float32)
            U_new[:(T-1)*4] = U_old[4:]
            U = self.on_gpu(U_new)
        print "Takeoff Succesfull, current state: "
        print self.state
        print
        #Now pursuit of the point begins
        p = np.array([1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)
        p_d = self.on_gpu(p)
        count = 0
        max_speed = 0
        direc = 1
        cycle = 0
        d = 1000
        speed = 500
        U_total = []
        while(count < 1000):
            print count
            U = self.rollouts(U, 500, 1.5, p_d, A_d, R_d)
            U_old = U.get()
            U_total.extend(U.get()[:4])
            U_total.extend(info[0][12:16])
            info = self.simulator(U_old.reshape(T,4), self.state,1)
            self.state= info[0]
            self.state_d = self.on_gpu(info[0])
            if (np.sqrt(info[0][6]**2 + info[0][7]**2 + info[0][8]**2) > max_speed):
                max_speed = np.sqrt(info[0][6]**2 + info[0][7]**2 + info[0][8]**2)
            X.extend(info[1])
            Y.extend(info[2])
            Z.extend(info[3])
            U_new = np.zeros(T*4, dtype = np.float32)
            U_new[:(T-1)*4] = U_old[4:]
            U = self.on_gpu(U_new)
            if (count < 1000):
                p[0] += .03
                p[1] += .03
            p_d = self.on_gpu(p)
            count += 1
            d = np.sqrt((np.sum(self.state[:3] - p[:3])**2))
            speed = np.sqrt(np.sum(self.state[6:9]*self.state[6:9]))
            print d
            print "-----------------------"
        print "Final Point Destination"
        print p
        print
        print "Final Quadrotor Destination"
        print self.state
        print 
        print "Maximum Speed"
        print max_speed
        #self.plotter(X,Y,Z)
        self.plotter(X,Y,Z,d2_plot=True)
        self.plot_controls(U_total)

    def plot_controls(self, U):
        length = len(U)/8
        U = np.array(U).reshape(length,8)
        l = np.arange(length)*self.dt
        fig = plt.figure()
        ax = fig.gca()
        #ax.plot(l, U[:,0])
        ax.plot(l, U[:,1])
        ax.plot(l, U[:,2])
        ax.plot(l, U[:,3])
        #ax.plot(l, U[:,7])
        plt.show()
       
    def draw_tree(self, ax, tree_info):
        x,y,h,r = tree_info
        U = np.arange(0,2*np.pi, .05)
        length = U.size
        V = np.arange(0,h,h/(1.0*length))
        X = np.zeros((length,length))
        Y = np.zeros((length,length))
        Z = np.zeros((length,length))
        for i in range(length):
            X[:,i] = np.cos(U)*r + x
            Y[:,i] = np.sin(U)*r + y
            Z[:,i] = V[i]
        ax.plot_surface(X,Y,Z)

    def plotter(self, X,Y,Z, d2_plot = False):
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        if (not d2_plot):
            fig = plt.figure()
            ax = fig.gca(projection = '3d')
            for i in range(len(self.forest)/4):
                self.draw_tree(ax, (self.forest[4*i:4*(i+1)]))
            ax.scatter(X,Y,Z)
            ax.set_xlim3d(0, 60)
            ax.set_ylim3d(0, 60)
        else:
            fig = plt.gcf()
            ax = plt.gca()
            for i in range(len(self.forest)/4):
                circ = plt.Circle((self.forest[4*i],self.forest[4*i + 1]), self.forest[4*i + 3], color = 'r')
                fig.gca().add_artist(circ)
            ax.set_xlim(-10,70)
            ax.set_ylim(-10,70)
            ax.plot(X,Y)
        plt.show()
 
        
    def simulator(self, U, state, steps):
        dt = self.dt
        T = self.T
        X = []
        Y = []
        Z = []
        s = state
        hover_speed = 3167.77
        w = np.zeros(4)
        max_speed = 9000
        min_speed = 1000
        vel = []
        for i in range(len(self.forest)):
            if (i % 4 == 0):
                self.forest[i] += .005
        self.forest_d = self.on_gpu(np.array(forest))
        for t in range(steps):
            noise = np.random.randn(6)*0
            for i in range(len(self.forest)/4):
                if np.sqrt((s[0] - self.forest[4*i + 0])**2 + (s[1] - self.forest[4*i + 1])**2) < self.forest[4*i + 3]+total_radius and self.forest[4*i + 2] > s[2]:     
                    for k in range(1):
                        print "HIT A TREE!"
                        print self.forest[4*i:4*i+3]
                        print s
                    break
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
            s[0] += dt*s[6]
            s[1] += dt*s[7]
            s[2] += dt*s[8]
            if (s[2] < 0):
                s[2] = 0
                s[8] = 0
            X.append(s[0])
            Y.append(s[1])
            Z.append(s[2])
            if (s[2] < 0):
                s[2] = 0
            s[3] += dt*(np.cos(s[4])*s[9] + np.sin(s[4])*s[11])
            s[4] += dt*(np.sin(s[4])*np.tan(s[3])*s[9] + s[10] - np.cos(s[4])*np.tan(s[3])*s[11])
            s[5] += dt*(-np.sin(s[4])/np.cos(s[3])*s[9] + np.cos(s[4])/np.cos(s[3])*s[11])
            F_1 = noise[3]*dt*.1*K_f + K_f * s[12]**2
            F_2 = noise[3]*dt*.1*K_f + K_f * s[13]**2
            F_3 = noise[3]*dt*.1*K_f + K_f * s[14]**2
            F_4 = noise[3]*dt*.1*K_f + K_f * s[15]**2
            F_sum = (F_1 + F_2 + F_3 + F_4)/total_mass
            s[6] += dt*F_sum*(np.cos(s[5])*np.sin(s[4]) + np.cos(s[4])*np.sin(s[5])*np.sin(s[3])) + (.2 + noise[0])*dt*.1
            s[7] += dt*F_sum*(np.sin(s[5])*np.sin(s[4]) - np.cos(s[5])*np.cos(s[4])*np.sin(s[3])) + noise[1]*dt*.1
            s[8] += dt*(F_sum*(np.cos(s[3])*np.cos(s[4])) - 9.81) + noise[2]*dt*.1
            s[9] += dt/Ix * (l*(F_2 - F_4) - s[10]*s[11]*(Iz - Iy))
            s[10] += dt/Iy * (l*(F_3 - F_1) - s[9]*s[11]*(Ix - Iz))
            M_1 = ((noise[4]*.1*K_m) + K_m)*s[12]**2
            M_2 = ((noise[4]*.1*K_m) + K_m)*s[13]**2
            M_3 = ((noise[4]*.1*K_m) + K_m)*s[14]**2
            M_4 = ((noise[4]*.1*K_m) + K_m)*s[15]**2
            s[11] += dt/Iz * (M_1 - M_2 + M_3 - M_4 - s[9]*s[10]*(Iy - Ix))
            s[12] += ((noise[5]*.1*motor_gain) + motor_gain)*dt*(w[0] - s[12])
            s[13] += ((noise[5]*.1*motor_gain) + motor_gain)*dt*(w[1] - s[13])
            s[14] += ((noise[5]*.1*motor_gain) + motor_gain)*dt*(w[2] - s[14])
            s[15] += ((noise[5]*.1*motor_gain) + motor_gain)*dt*(w[3] - s[15])
        return s, X, Y, Z

    def on_gpu(self,a):
        a = a.flatten()
        a = np.require(a, dtype = np.float32, requirements = ['A', 'O', 'W', 'C'])
        a_d = gpuarray.to_gpu(a)
        return a_d
    
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
    
    def func1(self):
        template = """
   
    #include <math.h>
    #include <stdio.h>
    
    #define T %d
    #define K %d
    #define dt %f
    #define STATE_DIM %d
    #define CONTROL_DIM %d
    #define Ix %f
    #define Iy %f
    #define Iz %f
    #define mass %f
    #define total_radius %f
    #define l %f
    #define gravity %f
    #define max_speed %f
    #define min_speed %f
    #define max_thrust %f
    #define min_thrust %f
    #define max_tilt %f
    #define min_tilt %f
    
    __device__ __constant__ float U_d[T*CONTROL_DIM];
    
    __device__ int get_crash(float* s, float* t, int num_trees) 
    {
       int crash = 0;
       if (s[2] < .1 && s[8] < -.1) {
          crash = 1;
       }
       int i;
       for (i = 0; i < num_trees; i++) {
           if (sqrt((s[0]-t[4*i + 0])*(s[0]-t[4*i + 0]) + (s[1]-t[4*i + 1])*(s[1]-t[4*i +1])) < t[4*i + 3]+total_radius && s[2] < t[4*i + 2]) {
           crash = 1;
           }
       }
       return crash;
    }

   /***********************************************************************************************
   Computes the immediate state cost of the system. 
   ************************************************************************************************/
    __device__ float get_state_cost(float* s, float* u, float* p, float* trees, float* A, float* R, int num_trees)
    { 
       float cost = 0;
       //Computes the penalties for the state cost
       float ground_penalty = 0;
       float tilt_penalty = 0;
       float crash_penalty = 0;
       if (s[2]*s[2] < .1 && s[8] < 0) {
          ground_penalty = 100;
       }
       if (fabs(s[3]) > 1.4) {
          tilt_penalty = 100;
       }
       float min = 10000;
       int i,j;
       for (j = 0; j < num_trees; j++) {
           float temp = sqrt((s[0]-trees[4*j + 0])*(s[0]-trees[4*j + 0]) + (s[1]-trees[4*j + 1])*(s[1]-trees[4*j +1]));
           if (temp < min) {
              min = temp;
           }
       }
       if (get_crash(s, trees, num_trees) == 1) {
          crash_penalty = 200;
       }
       cost += ground_penalty + tilt_penalty + crash_penalty + 10*exp(-min); 
       //Computes the control cost
       float control_cost = 0;
       for (i = 0; i < CONTROL_DIM; i++) {
          float temp = 0;
          for (j = 0; j < CONTROL_DIM; j++) {
             temp += R[i*CONTROL_DIM + j]*u[j];
          }
          control_cost += u[i]*temp;
       }
       cost += (control_cost);
       //Compute the state cost as a linear function of the current state (s) and goal state (p)
       float state_cost = 0;
       for (i = 0; i < STATE_DIM; i++) {
          state_cost += (s[i] - p[i])*A[i]*(s[i] - p[i]);
       }
       cost += state_cost;
       return cost;
    }

    __device__ float get_terminal_cost(float* s, float* p)
    {
       float cost = 0;
       return cost;
    }
   
    
    /*******************************************
    
    Kernel Function for Computing Rollouts
  
    *******************************************/
   
    __global__ void rollout_kernel(float* controls_d, float* state_costs_d, float* terminal_costs_d,
                                   float* du, float* init_state, float* trees, float* goal_state, 
                                   float* state_weights_d, float* control_weights_d, float* mech_params_d, 
                                   float var1, float var2, int num_trees)
    {
       //Get thread and block index
       int tdx = threadIdx.x;
       int bdx = blockIdx.x;  
       //Initialize local state
       __shared__ float R[CONTROL_DIM*CONTROL_DIM];
       __shared__ float A[STATE_DIM*STATE_DIM];
       __shared__ float p[STATE_DIM];
       float s[STATE_DIM];
       float u[CONTROL_DIM];
       float w[CONTROL_DIM];
       float hover_speed = 3167.77;
       float km = mech_params_d[0];
       float kf = mech_params_d[1];
       float motor_gain = mech_params_d[2];
       if (blockDim.x*bdx + tdx < K) {
          int i,j;
          if (tdx < STATE_DIM) {
             A[tdx] = state_weights_d[tdx];
             p[tdx] = goal_state[tdx];
             R[tdx] = control_weights_d[tdx];
          }
          if (tdx == 0) {
             for (i = 0; i < STATE_DIM; i++) {
                s[i] = init_state[i];
             }
          }
           __syncthreads();
          //Start the main program loop 
          for (i = 0; i < T; i++) {
             //Add exploration noise to the control commands
             for (j = 0; j < CONTROL_DIM; j++) {
                if (j %% 4 == 0) {
                   u[j] = U_d[i*CONTROL_DIM + j] + du[(blockDim.x*bdx + tdx)*T*CONTROL_DIM + i*CONTROL_DIM + j]*var1;
                }
                else {
                    u[j] = U_d[i*CONTROL_DIM + j] + du[(blockDim.x*bdx + tdx)*T*CONTROL_DIM + i*CONTROL_DIM + j]*var2;
                }
             }
             //Perform checks to see if any of the control commands violate the machines ability
             if (u[0] > max_thrust) {
                u[0] = max_thrust;
             }
             else if (w[0] < min_thrust) {
                u[0] = min_thrust;
             }
             if (u[1] > max_tilt) {
                u[1] = max_tilt;
             }
             else if (u[1] < min_tilt) {
                u[1] = min_tilt;
             }
             if (w[2] > max_tilt) {
                w[2] = max_tilt;
             }
             else if (w[2] < min_tilt) {
                w[2] = min_tilt;
             }
             if (w[3] > max_tilt) {
                w[3] = max_tilt;
             }
             else if (w[3] < min_tilt) {
                w[3] = min_tilt;
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
             //Check to see if the machine has crashed into a tree
             int crash = get_crash(s,trees, num_trees);
             //If the machine has not crashed update the state equations
             if (crash == 0) {
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
                float F_1 = kf * s[12]*s[12];
                float F_2 = kf * s[13]*s[13];
                float F_3 = kf * s[14]*s[14];
                float F_4 = kf * s[15]*s[15];
                float F_sum = (F_1 + F_2 + F_3 + F_4)/mass;
                s[6] += dt*F_sum*(cos(s[5])*sin(s[4]) + cos(s[4])*sin(s[5])*sin(s[3]));
                s[7] += dt*F_sum*(sin(s[5])*sin(s[4]) - cos(s[5])*cos(s[4])*sin(s[3]));
                s[8] += dt*(F_sum*(cos(s[3])*cos(s[4])) - gravity);
                s[9] += dt/Ix * (l*(F_2 - F_4) - s[10]*s[11]*(Iz - Iy));
                s[10] += dt/Iy * (l*(F_3 - F_1) - s[9]*s[11]*(Ix - Iz));
                float M_1 = km*s[12]*s[12];
                float M_2 = km*s[13]*s[13];
                float M_3 = km*s[14]*s[14];
                float M_4 = km*s[15]*s[15];
                s[11] += dt/Iz * (M_1 - M_2 + M_3 - M_4 - s[9]*s[10]*(Iy - Ix));
                s[12] += dt*motor_gain*(w[0] - s[12]);
                s[13] += dt*motor_gain*(w[1] - s[13]);
                s[14] += dt*motor_gain*(w[2] - s[14]);
                s[15] += dt*motor_gain*(w[3] - s[15]);
             }
             //Get State Costs
             float cost = get_state_cost(s, u, p, trees, A, R, num_trees);
             //Record the state costs
             state_costs_d[(blockDim.x*bdx + tdx)*T + i] = cost;
             //Record the control costs
             for (j = 0; j < CONTROL_DIM; j++) {
                controls_d[(blockDim.x*bdx + tdx)*T*CONTROL_DIM + i*CONTROL_DIM + j] = u[j];
             }
          }
          //Get and record the terminal costs
          if (blockDim.x*bdx + tdx < K) {
              float terminal_cost = get_terminal_cost(s, p);
              terminal_costs_d[blockDim.x*bdx + tdx] = terminal_cost;
          }
       }
    }
    """%(self.T, self.K, self.dt, state_dimensions, control_dimensions, Ix, Iy, Iz, total_mass, total_radius, l, gravity, max_rotor_speed, min_rotor_speed, max_thrust, min_thrust, max_tilt, min_tilt)
        mod = SourceModule(template)
        func = mod.get_function("rollout_kernel")
        U_d = mod.get_global("U_d")[0]
        return func, U_d

    
    def func2(self, T):
       template = """
 
    #include <math.h>
    #include <stdio.h>
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
      float lambda = -1.0/1000.0;
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

      if (tdy == 0 && x_ind < x_len && y_ind < y_len) {
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
    state= np.zeros(16, dtype=np.float32)
    T = 150
    K = 1000
    time_horizon = 3
    forest = []
    for i in range(1,20):
        for j in range(1,20):
            r = np.random.randn(2)
            forest.extend([3*i + r[0], 3*j + r[1], 10, .1])
    A = M_Pi_2(state, forest, K, time_horizon, T)
    A.multi_pi()
