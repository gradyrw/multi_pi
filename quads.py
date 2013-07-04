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
min_tilt = 4000.0
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
    def __init__(self, init_state, forest, K, time, T):
        #Store the initial stae values on both the CPU and GPU
        self.state = init_state
        self.state_d = self.on_gpu(state)
        #K is the number of rollouts to be performed per trial.
        self.K = K
        #T is the number of timesteps to be taken per rollout
        self.T = T
        #dt is the amount of time take from one time-step to another.
        self.dt = time/(1.0*T)
        #generator is a gpu function which quickly generates large amounts of normally
        #distributed random numbers.
        self.generator = XORWOWRandomNumberGenerator()
        #Initialize controls, state costs, and terminal costs as GPU arrays.
        self.controls_d = gpuarray.zeros(T*K*control_dimensions, dtype = np.float32)
        self.state_costs_d = gpuarray.zeros(T*K, dtype = np.float32)
        self.terminal_costs_d = gpuarray.zeros(K, dtype = np.float32)
        #Initialize the forest as a CPU and GPU array.
        self.forest = forest
        self.forest_d = self.on_gpu(forest)
        #Compile and store CUDA functions
        rollout_kernel, U_d = self.func1()
        cost_to_go = self.func2(T)
        reduction = self.func3()
        multiply = self.func4()
        self.funcs = rollout_kernel, cost_to_go, reduction, multiply, U_d

    def rollouts(self, U, var1, var2, goal_state_d, test = False):
        #Get an array of random numbers from CUDA. The numbers are
        #standard normal, the variance is changed later on in the rollout_kernel
        du = self.generator.gen_normal(self.K*self.T*(control_dimensions), np.float32)
        #Unpack CUDA functions from self.funcs, get addresses of current control
        #array in constant memory
        rollout_kernel, cost_to_go, reduction, multiply, U_d = self.funcs
        cuda.memcpy_dtod(U_d, U.ptr, U.nbytes)
        #Set blocksize and gridsize for rollout and cost-to-go kernels
        blocksize = (block_dim, 1, 1)
        gridsize = ((self.K - 1)//block_dim + 1, 1, 1)
        #Launch the kernel for simulating rollouts
        rollout_kernel(self.controls_d, self.state_costs_d, self.terminal_costs_d,
                       du, self.state_d, self.forest_d, goal_state_d, 
                       np.float32(motor_gain), np.float32(var1), np.float32(var2),
                       grid=gridsize, block=blocksize)
        cuda.Context.synchronize()
        d = self.terminal_costs_d.get()
        d = np.mean(d)
        c = self.state_costs_d.get()
        c = np.mean(c)
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
        reduction(self.state_costs_d, out_d, np.int32(K_int), np.int32(T_int),
                  grid=gridsize, block=blocksize)
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
        return out_d,d,c
    
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
        #Prepares the quadrotor for takeoff, p is the point it's aiming at.
        p = np.array([0,0,1.5],dtype=np.float32)
        p_d = self.on_gpu(p)
        #While loop for computing PI^2
        var = 250
        count = 0
        count_max = 500
        stop = False
        while (count < count_max and not stop):
            U,d,c = self.rollouts(U, 250, 0, p_d)
            if (d < 20):
                stop = True
            count += 1
        #Takeoff! Uses the pre-computed commands from the while loop
        info = self.plotter(U.get().reshape(T,4), state, T, plot=False)

        #Prints the state of the quadrotors after the takeoff commands have executed.
        print "After takeoff algorithm"
        p = np.array([0,0,2],dtype=np.float32)
        p_d = self.on_gpu(p)
        print info[0]
        #Records the new state into the class field
        self.state= info[0]
        self.state_d = self.on_gpu(info[0])
        #Records the takeoff trajectory
        X.extend(info[1])
        Y.extend(info[2])
        Z.extend(info[3])
        #Now hand over the commands to multi-pi to hover indefinately.
        U = np.zeros(T*4, dtype = np.float32)
        U = self.on_gpu(U)
        var = 500
        #Cancel Loop for now!!!!
        while((abs(self.state[8]) > .05 or abs(2 - self.state[2]) > .05)):
            U,d,c = self.rollouts(U, var, 0, p_d)
            U_old = U.get()
            info = self.plotter(U_old.reshape(T,4), self.state, 1, plot=False)
            self.state= info[0]
            self.state_d = self.on_gpu(info[0])
            #Prints some info about the curent state
            print (self.state[2], self.state[8], self.state[12], np.mean(U.get()))
            print
            X.extend(info[1])
            Y.extend(info[2])
            Z.extend(info[3])
            U_new = np.zeros(T*4, dtype = np.float32)
            U_new[:(T-1)*4] = U_old[4:]
            U = self.on_gpu(U_new)
        #Now move to the point (2,2,2)
        print "====================================="
        count = 0
        d = 21
        """
        while(count < 500):
            U,d,c = self.rollouts(U, 500, 1.5, p_d, 1)
            print p_d, self.state_d
            print
            print count
            print d,c
            print U
            print 
            if (np.mean(p_d.get()) == 0):
                break
            count += 1
        print "======================================"
        self.plotter(U.get().reshape(T,4), self.state,T,plot=True)
        """
        p = np.array([5,5,5], dtype=np.float32)
        p_d_test = self.on_gpu(p)
        count = 0
        t = 0
        while(count < 1000):
            count += 1
            if count % 500 == 0:
                p = np.array([5,5,5])
                p_d_test = self.on_gpu(p)
            if count % 1000 == 0:
                p = np.array([5,5,5])
                p_d_test = self.on_gpu(p)
            print count
            #print "++++++++++++++++++++++++++++++++++++++++"
            #print p_d_test
            U,d,c = self.rollouts(U, 500, 1.5, p_d_test)
            #print p_d_test
            #print "========================================"
            #print p_d_test
            #print "ttttttttttttttttttttttttttttttttttttttttttt"
            U_old = U.get()
            info = self.plotter(U_old.reshape(T,4), self.state,1,trees = [1,1,10,.25,2,4,10,.25, 4,1,10,.25,5,3,10,.25, 3,4,10,.25], plot=False)
            #print p_d_test
            #print "-----------------------------------------"
            self.state= info[0]
            self.state_d = self.on_gpu(info[0])
            #print info[0], info[1], info[2]
            X.extend(info[1])
            Y.extend(info[2])
            Z.extend(info[3])
            U_new = np.zeros(T*4, dtype = np.float32)
            U_new[:(T-1)*4] = U_old[4:]
            U = self.on_gpu(U_new)
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        self.draw_tree(ax, (1,1,10,.25))
        self.draw_tree(ax, (2,4,10,.25))
        self.draw_tree(ax, (4,1,10,.25))
        self.draw_tree(ax, (5,3,10,.25))
        self.draw_tree(ax, (3,4,10,.25)) 
        ax.scatter(X,Y,Z)
        ax.set_xlim3d(0, 10)
        ax.set_ylim3d(0, 10)
        plt.show()

    def draw_tree(self, ax, tree_info):
        x,y,h,r = tree_info
        U = np.arange(0,2*np.pi, .05)
        length = U.size
        #print length
        #print h
        V = np.arange(0,h,h/(1.0*length))
        print V.size
        X = np.zeros((length,length))
        Y = np.zeros((length,length))
        Z = np.zeros((length,length))
        for i in range(length):
            X[:,i] = np.cos(U)*r + x
            Y[:,i] = np.sin(U)*r + y
            Z[:,i] = V[i]
        ax.plot_surface(X,Y,Z)
        
    def plotter(self, U, state, jumps, trees = [], plot = False):
        dt = self.dt
        T = self.T
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        X = []
        Y = []
        Z = []
        s = state
        hover_speed = 3167.77
        w = np.zeros(4)
        max_speed = 9000
        min_speed = 1000
        vel = []
        for t in range(jumps):
            for i in range(len(trees)/4):
                if np.sqrt((s[0] - trees[4*i + 0])**2 + (s[1] - trees[4*i + 1])**2) < trees[4*i + 3] and trees[4*i + 2] > s[2]:     
                    for k in range(1):
                        print "HIT A TREE!"
                        print trees[4*i:4*i+3]
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
           
            #print 
            #print "----------------------------------------"
            #print w
            #print
            #if (abs(s[3]) > 1):
            #    print "DANGER"
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
            F_1 = K_f * s[12]**2
            F_2 = K_f * s[13]**2
            F_3 = K_f * s[14]**2
            F_4 = K_f * s[15]**2
            F_sum = (F_1 + F_2 + F_3 + F_4)/total_mass
            s[6] += dt*F_sum*(np.cos(s[5])*np.sin(s[4]) + np.cos(s[4])*np.sin(s[5])*np.sin(s[3]))
            s[7] += dt*F_sum*(np.sin(s[5])*np.sin(s[4]) - np.cos(s[5])*np.cos(s[4])*np.sin(s[3]))
            s[8] += dt*(F_sum*(np.cos(s[3])*np.cos(s[4])) - 9.81)
            s[9] += dt/Ix * (l*(F_2 - F_4) - s[10]*s[11]*(Iz - Iy))
            s[10] += dt/Iy * (l*(F_3 - F_1) - s[9]*s[11]*(Ix - Iz))
            M_1 = K_m*s[12]**2
            M_2 = K_m*s[13]**2
            M_3 = K_m*s[14]**2
            M_4 = K_m*s[15]**2
            s[11] += dt/Iz * (M_1 - M_2 + M_3 - M_4 - s[9]*s[10]*(Iy - Ix))
            s[12] += motor_gain*dt*(w[0] - s[12])
            s[13] += motor_gain*dt*(w[1] - s[13])
            s[14] += motor_gain*dt*(w[2] - s[14])
            s[15] += motor_gain*dt*(w[3] - s[15])

        if (plot == True):
            X = np.array(X)
            Y = np.array(Y)
            Z = np.array(Z)
            ax.scatter(X,Y,Z)
            ax.set_xlim3d(-10, 10)
            ax.set_ylim3d(-10, 10)
            plt.show()
            print s
        return s, X, Y, Z
    
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
           if (sqrt((s[0]-t[4*i + 0])*(s[0]-t[4*i + 0]) + (s[1]-t[4*i + 1])*(s[1]-t[4*i +1])) < t[4*i + 3] && s[2] < t[4*i + 2]) {
           crash = 1;
           }
       }
       return crash;
    }

    __device__ float get_state_cost(float* s, float* u, float* p, float* trees, int num_trees)
    { 
       float cost = 0;
       float ground_penalty = 0;
       float tilt_penalty = 0;
       float crash_penalty = 0;
       cost = 1*( (u[0]/1000.0)*(u[0]/1000.0) + (u[1]*u[1]) + (u[2]*u[2])+(u[3]*u[3]) )
              + 1*( (s[3]*s[3]) + (s[4]*s[4]) + (s[5]*s[5]) ) 
              + 5*( (s[0]-p[0])*(s[0]-p[0]) + (s[1]-p[1])*(s[1]-p[1]) + (s[2]-p[2])*(s[2]-p[2]) )
              + 1*( (s[6]*s[6]) + (s[7]*s[7]) + (s[8]*s[8]) )
              + 1*( (s[9]*s[9]) + (s[10]*s[10]) + (s[11]*s[11]) );
/       if (s[2]*s[2] < .1 && s[8] < 0) {
          ground_penalty = 100;
       }
       cost += ground_penalty;
       if (fabs(s[3]) > 1.4 || fabs(s[4]) > 1.4) {
          tilt_penalty = 0;
       }
       if (get_crash(s, trees, num_trees) == 1) {
          crash_penalty = 1000;
       }
       cost += tilt_penalty + crash_penalty + tilt_penalty;
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
                                   float motor_gain, float var1, float var2)
    {
       int num_trees = 5;
       //Get thread and block index
       int tdx = threadIdx.x;
       int bdx = blockIdx.x;
       //Initialize local state
       float s[STATE_DIM];
       float p[STATE_DIM];
       float u[CONTROL_DIM];
       float w[CONTROL_DIM];
       float hover_speed = 3167.77;
       float km = .00015;
       float kf = .00611;
       int i,j;
       for (i = 0; i < STATE_DIM; i++) {
          s[i] = init_state[i];
       }
       for (i = 0; i < STATE_DIM; i++) {
          p[i] = goal_state[i];
       }
       if (blockDim.x*bdx + tdx < K) {
          for (i = 0; i < T; i++) {
             for (j = 0; j < CONTROL_DIM; j++) {
                if (j %% 4 == 0) {
                   u[j] = U_d[i*CONTROL_DIM + j] + du[(blockDim.x*bdx + tdx)*T*CONTROL_DIM + i*CONTROL_DIM + j]*var1;
                }
                else {
                    u[j] = U_d[i*CONTROL_DIM + j] + du[(blockDim.x*bdx + tdx)*T*CONTROL_DIM + i*CONTROL_DIM + j]*var2;
                }
             }
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
             int crash = get_crash(s,trees, num_trees);
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
             }
             //Get State Costs
             float cost = get_state_cost(s, u, p, trees, num_trees);
             //Add costs into state_costs 
             state_costs_d[(blockDim.x*bdx + tdx)*T + i] = cost;
             //Add controls
             for (j = 0; j < CONTROL_DIM; j++) {
                controls_d[(blockDim.x*bdx + tdx)*T*CONTROL_DIM + i*CONTROL_DIM + j] = u[j];
             }
          }
          //Get Terminal Costs
          if (blockDim.x*bdx + tdx < K) {
              float terminal_cost = get_terminal_cost(s, p);
              terminal_costs_d[blockDim.x*bdx + tdx] = terminal_cost;
          }
       }
    }
    """%(self.T, self.K, self.dt, state_dimensions, control_dimensions, Ix, Iy, Iz, total_mass, l, gravity, max_rotor_speed, min_rotor_speed, max_thrust, min_thrust, max_tilt, min_tilt)
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

    def on_gpu(self,a):
        a = a.flatten()
        a = np.require(a, dtype = np.float32, requirements = ['A', 'O', 'W', 'C'])
        a_d = gpuarray.to_gpu(a)
        return a_d

if __name__ == "__main__":
    state= np.zeros(16, dtype=np.float32)
    state[12:] = 3167.77
    state[2] = 0
    T = 150
    K = 1000
    U = np.zeros(T*4, dtype = np.float32)
    var1 = 100
    var2 = 1.5
    var11 = var1
    var22 = var2
    forest = np.array([1,1,10,.25,2,4,10,.25, 4,1,10,.25,5,3,10,.25, 3,4,10,.25], dtype = np.float32)
    p = np.array([0,0,1.5])
    #print p
    A = M_Pi_2(state, forest, K, 3, T)
    p_d = A.on_gpu(p)
    U = A.on_gpu(U)
  
    """
    count = 0
    d_orig = 0
    d = 1
    stop = False
    count_max = 1000
    #start = time.clock()
    while (count < count_max and not stop):
        start = time.clock()
        U,d = A.rollouts(U, var11, var2*0, p_d, 1)
        if count == 0:
            d_orig = d
        cost = d
        d = (d/d_orig)
        var11 = var1*d
        var22 = var2*d
        print time.clock() - start
        if (cost < 5):
            stop = True
        count += 1
    #print time.clock() - start
    print count
    state= A.plotter(U.get().reshape(T,4), state, jumps, plot = True)
    """
    A.multi_pi()
