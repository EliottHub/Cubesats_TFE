import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from Functions import *


# Determines the height of local maxima within a specific range
def find_second_local_maximum(F, p):
    maxima = []
    start = np.where(p >= 179.2)[0][0]
    end = np.where(p <= 180.6)[0][-1]

    for i in range(start + 1, end):
        if F[i] > F[i - 1] and F[i] > F[i + 1]:
            maxima.append(F[i])

    if len(maxima) < 2:
        return None

    maxima.sort(reverse=True)
    #print(maxima)
    return maxima[1]


def pos_sat(t,w):

    Opti_Spiral = 1
    Opti_Single_Orbit = 0
    Opti_Sym_Orbit = 0

    min = 200

    num_points = 3600*100
    T = 2*np.pi/w #period
    t_Cubesat = np.linspace(0, T, num_points)

    if (Opti_Spiral):

        colors = ['r', 'm', 'b', 'g','orange']
        count=0 

        for i in np.arange(2,5,1):
            for j in np.arange(4, 8, 1):
                for k in np.arange(9, 14, 1):
                    for u in np.arange(15, 19, 1):
                        for v in np.arange(19, 25, 1):
                            for r in np.arange(25, 30, 1):

                                phi0_list = [0,120,240]
                                dephasage, index = 0,1
                                x0_list = [i,j,k,u,v,r,29,30,33,35]  

                                num_sat = len(phi0_list)*len(x0_list)+1 
                                pos_sat = np.zeros((num_sat,3))

                                for x0 in x0_list:
                                    for phi0, color in zip(phi0_list, colors):

                                        x0_dot = 0.0
                                        y0 = 2*x0_dot/w
                                        y0_dot = -2*w*x0
                                        z0,z0_dot,theta0 = 0,0,0

                                        x,y,z = relative_3Dmotion(t_Cubesat[t],x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0-dephasage),np.radians(theta0),w) 
                                        pos_sat[index,:] = np.array([x,y,z])
                                        index+=1
                                    dephasage+=230/len(x0_list) #deg 

                                acc = 0.01
                                phi_min = 170
                                phi_max = 190
                                p = np.arange(phi_min,phi_max+acc,acc)

                                # acc = 0.01
                                # min = 170
                                # max = 190
                                # p = np.arange(min,max+acc,acc)

                                F = radiation_pattern_partial(pos_sat, p, acc)
                                F = 10 * np.log10(F[:, 0])

                                # if i==180:
                                #     plt.plot(p,F)
                                #     plt.ylim(-15,0)
                                #     plt.xlim(178,182)

                                min_local = find_second_local_maximum(F, p)

                                if min_local<min:
                                    min = min_local
                                    i_ideal = i
                                    j_ideal = j
                                    k_ideal = k
                                    u_ideal = u
                                    v_ideal = v
                                    r_ideal = r
                                
                                count+=1
                                print(count)

        print('i_ideal =',i_ideal)
        print('j_ideal =',j_ideal)
        print('k_ideal =',k_ideal)
        print('u_ideal =',u_ideal)
        print('v_ideal =',v_ideal)
        print('r_ideal =',r_ideal)
        print('min_local =',min)
        
        plt.show()  



    if (Opti_Sym_Orbit):

        count=0 

        for i in np.arange(5, 50, 1):
            for j in np.arange(10, 200, 10):

                x0_dot = 0.
                x0 = i
                y0_dot = -2*w*x0
                z0 = 0
                z0_dot = 0.0
                theta0 = 0
                C = np.sqrt( (3*x0+2*y0_dot/w)**2 + (x0_dot/w)**2 )
                y0_list = [(2*C + j) + 2*x0_dot/w , -(2*C + j) + 2*x0_dot/w]
                phi0_list = np.linspace(0,360,15)

                index = 1

                num_sat = len(y0_list)*len(phi0_list)+1  
                pos_sat = np.zeros((num_sat,3))

                for y0 in y0_list:
                    for phi0 in phi0_list:
                        x,y,z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w) 
                        pos_sat[index,:] = np.array([x,y,z])
                        index+=1

                acc = 0.01
                phi_min = 170
                phi_max = 190
                p = np.arange(phi_min,phi_max+acc,acc)

                F = radiation_pattern_partial(pos_sat, p, acc)
                F = 10 * np.log10(F[:, 0])

                min_local = find_second_local_maximum(F, p)

                if min_local<min:
                    min = min_local
                    i_ideal = i
                    j_ideal = j
                
                count+=1
                print(count)

        print('i_ideal =',i_ideal)
        print('j_ideal =',j_ideal)
        print('min_local =',min)
    


    if (Opti_Single_Orbit):

        num_sat_list = np.arange(21,33,1)  
        min = 20.0
        num_sat_ideal = 0
        count=0
        
        for j in np.arange(69,70,1):
            for num_sat in num_sat_list:

                dephasage, index = 0,1
                pos_sat = np.zeros((num_sat,3))

                for i in range(num_sat-1):

                    phi0 = 0.0
                    x0 = j
                    x0_dot = 0.0
                    y0 = 2*x0_dot/w
                    y0_dot = -2*w*x0
                    z0 = 0.0
                    z0_dot = 0.0
                    theta0 = -(90+phi0+dephasage)

                    x,y,z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0-dephasage),np.radians(theta0),w) 
                    pos_sat[index,:] = np.array([x,y,z])
                    index+=1
                    dephasage+=360/(num_sat-1) #deg 
                
                    acc = 0.01
                    phi_min = 170
                    phi_max = 190
                    p = np.arange(phi_min,phi_max+acc,acc)

                    F = radiation_pattern_partial(pos_sat, p, acc)
                    F = 10 * np.log10(F[:, 0])

                    min_local = find_second_local_maximum(F, p)

                    try:
                        if min_local is not None and min_local < min:
                            min = min_local
                            i_ideal = i
                            num_sat_ideal = num_sat
                    except TypeError:
                        pass
                    
                    count+=1
                    print(count)

        print('i_ideal =',i_ideal)
        print('num_sat_ideal =',num_sat_ideal)
        print('min_local =',min)

    return pos_sat


w = angular_rate(400)
pos_sat(00*100,w)

width = 10 #km width of the specular zone
alpha = glistening_angle(width)
plt.axvline(x=180+alpha,color="green",linestyle='--',label="Glistening zone")
plt.axvline(x=180-alpha,color="green",linestyle='--')











