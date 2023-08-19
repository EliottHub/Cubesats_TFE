import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
from SpecularPoint import *
from PIL import Image
from numba import jit

Re = 6371
#mu = 3.986e14 #m3/s2
mu = 398600.5 #km3/s2
r_Cubesat = Re + 400 #km
r_GNSS = Re + 23222

w_Cubesat = np.sqrt(mu/r_Cubesat**3) #rad/s
w_GNSS = np.sqrt(mu/r_GNSS**3)

#T_min = 2*np.pi/w/60
#print('Period (min) =',T_min)

def circular_orbit(a, w, inclination, raan, argp, t):
    inclination = np.deg2rad(inclination)
    raan = np.deg2rad(raan)  # right ascension of the ascending node
    argp = np.deg2rad(argp)  # argument of perigee

    x = a * np.cos(w * t)
    y = a * np.sin(w * t)
    z = np.zeros_like(t)

    # Rotation about the Z axis: aligns the orbit plane with the equatorial plane
    R3_W = np.array([[np.cos(raan), np.sin(raan), 0], [-np.sin(raan), np.cos(raan), 0], [0, 0, 1]])
    # Rotation around the X-axis: tilts the orbit plane by the given tilt angle
    R1_i = np.array([[1, 0, 0], [0, np.cos(inclination), np.sin(inclination)], [0, -np.sin(inclination), np.cos(inclination)]])
    # Z-axis rotation: rotates the orbit around the Z-axis according to the argument of the periapsis and the ascending node
    R3_w = np.array([[np.cos(argp), np.sin(argp), 0], [-np.sin(argp), np.cos(argp), 0], [0, 0, 1]])

    R = R3_W.dot(R1_i).dot(R3_w)
    orbit = np.vstack((x, y, z))
    velocity = np.vstack((-a*w*np.sin(w*t), a*w*np.cos(w*t), np.zeros_like(t)))

    return R.dot(orbit), R.dot(velocity)

def relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w):

    y_c0 = y0 - (2*x0_dot / w)
    y_c_dot = -6*w*x0 - 3*y0_dot
    x_c = -2*y_c_dot / (3*w)
    z_c = z0

    C = np.sqrt( (3*x0+2*y0_dot/w)**2 + (x0_dot/w)**2 )
    D = np.sqrt((z0_dot/w)**2 + z0**2)
    
    x = x_c  + C*np.sin(w*t + phi0)
    y = y_c0 + y_c_dot*t + 2*C*np.cos(w*t + phi0) 
    z = D*np.cos(w*t - theta0)

    return x,y,z 

@jit(nopython=True)
def ground_spot(specular_coord, sat_center_coord, pos_sat, width, integral):
   
    F = np.zeros((int(width/10)+1,int(width/10)+1),dtype="complex") #Radiation pattern
    lambdaa = 0.2
    k = 2*np.pi/lambdaa
    
    u = np.zeros((int(width/10)+1,int(width/10)+1,3))
    u0 = (specular_coord - sat_center_coord)/np.linalg.norm(specular_coord - sat_center_coord) #beam steering method, u0: vector pointing from the center of the array to the specular point
    ground_coord = np.zeros((int(width/10)+1,int(width/10)+1,3))

    num_sat = pos_sat.shape[0]
    x_grid = int(width/10) + 1
    y_grid = int(width/10) + 1
    
    for i in range(x_grid):
        for j in range(y_grid):
            ground_coord[i,j,0] = specular_coord[0] + 10*(i - int(width/20))
            ground_coord[i,j,1] = specular_coord[1] + 10*(j - int(width/20))
            ground_coord[i,j,2] = specular_coord[2] + 0
            
            u[i,j,0] = ground_coord[i,j,0] - sat_center_coord[0]
            u[i,j,1] = ground_coord[i,j,1] - sat_center_coord[1]
            u[i,j,2] = ground_coord[i,j,2] - sat_center_coord[2]
            
            for n in range(num_sat):
                rho = pos_sat[n,:] - sat_center_coord
                u[i,j,:] /= np.linalg.norm(u[i,j,:])
                r = 1#np.linalg.norm(specular_coord-sat_center_coord)
                F[i,j] += (1/r)*np.exp(1j*k*np.dot((u[i,j,:]-u0),rho))
                
    F = np.abs(F)
    F = F/np.amax(F)

    D = (4*np.pi/integral)*(F**2)
    D = 10*np.log10(D)

    G = 1*D
    
    return G,F

@jit(nopython=True)
def Radiation_Pattern(antenna_pos, sat_center_coord, specular_coord, acc):

    phi = int(360/acc)
    theta = int(180/acc)
    lambdaa = 0.2 #m
    num_sat = antenna_pos.shape[0]
    
    k = 2*np.pi/lambdaa
    u = np.zeros((phi,theta,3)) #on va créer un u(x,y,z) pour tous les (phi, theta)
    F = np.zeros((phi,theta),dtype=np.complex128) #chaque (phi,theta) aura sa valeur de F

    u0 = (specular_coord-sat_center_coord)/np.linalg.norm(specular_coord-sat_center_coord)

    for i in range(phi): #boucle sur les phi
        for l in range(theta): #boucle sur les theta
            u[i,l,:] = [np.sin(i*np.pi/theta)*np.cos(l*np.pi/theta),np.sin(i*np.pi/theta)*np.sin(l*np.pi/theta),np.cos(i*np.pi/theta)] #unit (rho=1) director vector u in cartesian coord.
            for j in range(num_sat): #boucle sur les antennes/satellites
                rho = antenna_pos[j,:]-sat_center_coord
                u[i,l,:] /= np.linalg.norm(u[i,l,:])
                r = 1 #np.linalg.norm(specular_coord-sat_center_coord)
                F[i,l] = F[i,l] + (1/r)*np.exp(1j*k*np.dot(u[i,l,:],rho))*np.exp(-1j*k*np.dot(u0,rho))  #(additional phase shift)  
                if np.array_equal(u[i,l,:], u0):
                    print("F_num_sat1 =", F[i,l])
                
    F = np.abs(F)
    F = F/np.amax(F) #normalize values in array F

    # Directivity 
    t = np.deg2rad(np.arange(0,180,acc))
    p = np.deg2rad(np.arange(0,360,acc))
    sin_theta = np.sin(t)
    integral = 0
    for i in range(theta):
        for j in range(phi):
            integral = integral + (np.pi/theta)*(2*np.pi/phi) * F[j,i]**2 * sin_theta[i]
    D = (4*np.pi/integral)*(F**2)
    print("integral=", integral)

    return F,D,integral



motion_Y_shape = 0 
motion_X_shape = 0 
animation_ground_spot = 0
animation_specular_point = 1




#----------------------- Absolute motion of Cubesats in Y-shape (x-y-z plan)---------------------- 
if motion_Y_shape == 1:
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        # Plot chief satellite
        #s0, = ax.plot([], [], [], 'ko')

        ax.set_xlim(-7000, 7000)
        ax.set_ylim(-7000, 7000)
        ax.set_zlim(-7000, 7000)
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')
        #plt.title('Spiral formation')

        # Define the ciruclar chief orbit
        a = 6371 + 400  # semi-major axis in km
        inclination = 0 # determines how tilted the orbit is relative to the Earth's equator (deg)
        raan = 0
        argp = 0
        num_points = 3601
        T = 2*np.pi/w_Cubesat #period
        t = np.linspace(0, T, num_points)
        circular_orbit_Cubesat = circular_orbit(a, w_Cubesat, inclination, raan, argp, t)
        ax.plot(circular_orbit_Cubesat[0][0], circular_orbit_Cubesat[0][1],circular_orbit_Cubesat[0][2], color='g',label='Absolut Chief Orbit', markersize=1)


        y0_list = np.linspace(1,15,15) 
        y0_list = np.array(y0_list) * 200
        y0_list2 = np.linspace(-1,-15,15) #pour le X
        y0_list2 = np.array(y0_list2) * 200

        lines = []

        t_obs = np.linspace(0, 3600, 100)

        # Create the s values to set the data
        for y0 in range(len(y0_list)+len(y0_list2)):
            label = "s" + str(len(lines) + 1)
            line, = ax.plot([], [], [], 'bo', label=label)
            lines.append(line)

        # Un de plus pour le satellite principal
        label = "s" + str(len(lines) + 1)
        line, = ax.plot([], [], [], 'ko', label=label)
        lines.append(line)

        print(len(lines))


        for i in t_obs:
            i = int(i)
            Time = t[i]
            print(i)

            index_lines = 0
            x = circular_orbit_Cubesat[0][0][i]
            y = circular_orbit_Cubesat[0][1][i]
            z = circular_orbit_Cubesat[0][2][i]
            lines[index_lines].set_data(x,y)
            lines[index_lines].set_3d_properties(-z)
            #lines[index_lines].set_markersize(1)
            index_lines += 1

            # Absolute poisiton of the deputy satellites
            y0_list_neg = np.linspace(-10,-1,10)
            y0_list_neg = np.array(y0_list_neg) * 500
            y0_list_pos = np.linspace(1,20,20) 
            y0_list_pos = np.array(y0_list_pos) * 200
            y0_list = np.concatenate((y0_list_neg, y0_list_pos))

            x0_dot,x0,y0_dot,z0_dot = 0.0,0.0,0.0,0.0
            z0 = 0.0
            phi0 = 0.0
            theta0 = 0.0
            count=0

            for y0 in y0_list:
                if y0>=1:
                    z0+=200.
                    x_rel, y_rel, z_rel = relative_3Dmotion(Time,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w_Cubesat)
                
                else: 
                    x_rel, y_rel, z_rel = relative_3Dmotion(Time,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w_Cubesat)
                
                # Position du premier satellite
                pos1 = np.array([circular_orbit_Cubesat[0][0][i], circular_orbit_Cubesat[0][1][i], circular_orbit_Cubesat[0][2][i]])
                vit1 = np.array([circular_orbit_Cubesat[1][0][i], circular_orbit_Cubesat[1][1][i], circular_orbit_Cubesat[1][2][i]])

                # Vecteur pointant du centre de la Terre vers le premier satellite
                earth_to_sat1 = pos1/np.linalg.norm(pos1)
                normal_dir = np.cross(earth_to_sat1, vit1)
                tangential_dir = np.cross(pos1, normal_dir)

                # Vector normalization
                tangential_dir /= np.linalg.norm(tangential_dir)
                normal_dir /= np.linalg.norm(normal_dir)

                # Rotation matrix
                R = np.array([earth_to_sat1, tangential_dir, normal_dir]).T

                pos_rel = np.vstack((x_rel, y_rel, z_rel))
                x_rel = R.dot(pos_rel)[0]
                y_rel = R.dot(pos_rel)[1]
                z_rel = R.dot(pos_rel)[2]
                
                x = x_rel + circular_orbit_Cubesat[0][0][i]
                y = y_rel + circular_orbit_Cubesat[0][1][i]
                z = z_rel + circular_orbit_Cubesat[0][2][i]

                if y0 >= 1:
                    if count % 2 != 0:
                        lines[index_lines].set_data(x,y)
                        lines[index_lines].set_3d_properties(-z)
                        lines[index_lines].set_markersize(1)
                        index_lines += 1
                    else:
                        lines[index_lines].set_data(x,y)
                        lines[index_lines].set_3d_properties(z)
                        lines[index_lines].set_markersize(1)
                        index_lines += 1
                    count += 1
                else:
                    lines[index_lines].set_data(x,y)
                    lines[index_lines].set_3d_properties(z)
                    lines[index_lines].set_markersize(1)
                    index_lines += 1


            writer.grab_frame()


    # create animation object
    ani = animation.FuncAnimation(fig, animate, frames=1, interval=0.5)

    # create PillowWriter object with higher fps value
    writer = PillowWriter(fps=1000)

    # save animation with PillowWriter
    ani.save('Y_3DMotion.gif', writer=writer)










#----------------------- Absolute motion of Cubesats in X-shape (x-y-z plan)---------------------- 
if motion_X_shape == 1:
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        # Plot chief satellite
        #s0, = ax.plot([], [], [], 'ko')

        ax.set_xlim(-7000, 7000)
        ax.set_ylim(-7000, 7000)
        ax.set_zlim(-7000, 7000)
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')
        #plt.title('Spiral formation')

        # Define the ciruclar chief orbit
        a = 6371 + 400  # semi-major axis in km
        inclination = 90 # determines how tilted the orbit is relative to the Earth's equator (deg)
        raan = 0
        argp = 0
        num_points = 3601
        T = 2*np.pi/w_Cubesat #period
        t = np.linspace(0, T, num_points)
        circular_orbit_points = circular_orbit(a, w_Cubesat, inclination, raan, argp, t)
        ax.plot(circular_orbit_points[0][0], circular_orbit_points[0][1],circular_orbit_points[0][2], color='g',label='Absolut Chief Orbit', markersize=1)


        y0_list = np.linspace(1,15,15) 
        y0_list = np.array(y0_list) * 200
        y0_list2 = np.linspace(-1,-15,15) #pour le X
        y0_list2 = np.array(y0_list2) * 200

        lines = []

        t_obs = np.linspace(0, 3600, 100)

        # Create the s values to set the data
        for y0 in range(len(y0_list)+len(y0_list2)):
            label = "s" + str(len(lines) + 1)
            line, = ax.plot([], [], [], 'bo', label=label)
            lines.append(line)
        print(len(lines))

        #with writer.saving(fig, "Y-3DMotion.gif", 100): #enlever cette ligne permet de ne pas avoir de doublon

        for i in t_obs:
            i = int(i)
            Time = t[i]
            print(i)

            index_lines = 0
            x0_dot,x0,y0_dot,z0_dot = 0.0,0.0,0.0,0.0
            z0 = 0.0
            phi0 = 0.0
            theta0 = 0.0
            count=0

            for y0 in y0_list:
                if y0>=1:
                    z0+=200.
                    x_rel, y_rel, z_rel = relative_3Dmotion(Time,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w_Cubesat)
                
                else: 
                    x_rel, y_rel, z_rel = relative_3Dmotion(Time,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w_Cubesat)
                
                # Position du premier satellite
                pos1 = np.array([circular_orbit_points[0][0][i], circular_orbit_points[0][1][i], circular_orbit_points[0][2][i]])
                vit1 = np.array([circular_orbit_points[1][0][i], circular_orbit_points[1][1][i], circular_orbit_points[1][2][i]])

                # Vecteur pointant du centre de la Terre vers le premier satellite
                earth_to_sat1 = pos1/np.linalg.norm(pos1)
                normal_dir = np.cross(earth_to_sat1, vit1)
                tangential_dir = np.cross(pos1, normal_dir)

                # Vector normalization
                tangential_dir /= np.linalg.norm(tangential_dir)
                normal_dir /= np.linalg.norm(normal_dir)

                # Rotation matrix
                R = np.array([earth_to_sat1, tangential_dir, normal_dir]).T

                pos_rel = np.vstack((x_rel, y_rel, z_rel))
                x_rel = R.dot(pos_rel)[0]
                y_rel = R.dot(pos_rel)[1]
                z_rel = R.dot(pos_rel)[2]
                
                x = x_rel + circular_orbit_points[0][0][i]
                y = y_rel + circular_orbit_points[0][1][i]
                z = z_rel + circular_orbit_points[0][2][i]

                if y0 >= 1:
                    if count % 2 != 0:
                        lines[index_lines].set_data(x,-y)
                        lines[index_lines].set_3d_properties(z)
                        lines[index_lines].set_markersize(1)
                        index_lines += 1
                    else:
                        lines[index_lines].set_data(x,y)
                        lines[index_lines].set_3d_properties(z)
                        lines[index_lines].set_markersize(1)
                        index_lines += 1
                    count += 1
                else:
                    lines[index_lines].set_data(x,y)
                    lines[index_lines].set_3d_properties(z)
                    lines[index_lines].set_markersize(1)
                    index_lines += 1
            

            z0 = 0
            count=0

            for y0 in y0_list2:
                z0+=200.
                x_rel,y_rel,z_rel = relative_3Dmotion(Time,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w_Cubesat) 


                # Position du premier satellite
                pos1 = np.array([circular_orbit_points[0][0][i], circular_orbit_points[0][1][i], circular_orbit_points[0][2][i]])
                vit1 = np.array([circular_orbit_points[1][0][i], circular_orbit_points[1][1][i], circular_orbit_points[1][2][i]])

                # Vecteur pointant du centre de la Terre vers le premier satellite
                earth_to_sat1 = pos1/np.linalg.norm(pos1)
                normal_dir = np.cross(earth_to_sat1, vit1)
                tangential_dir = np.cross(pos1, normal_dir)

                # Vector normalization
                tangential_dir /= np.linalg.norm(tangential_dir)
                normal_dir /= np.linalg.norm(normal_dir)

                # Rotation matrix
                R = np.array([earth_to_sat1, tangential_dir, normal_dir]).T

                pos_rel = np.vstack((x_rel, y_rel, z_rel))
                x_rel = R.dot(pos_rel)[0]
                y_rel = R.dot(pos_rel)[1]
                z_rel = R.dot(pos_rel)[2]
                
                x = x_rel + circular_orbit_points[0][0][i]
                y = y_rel + circular_orbit_points[0][1][i]
                z = z_rel + circular_orbit_points[0][2][i]

                if count % 2 != 0:
                    lines[index_lines].set_data(x,-y)
                    lines[index_lines].set_3d_properties(z)
                    lines[index_lines].set_markersize(1)
                    index_lines += 1
                else:
                    lines[index_lines].set_data(x,y)
                    lines[index_lines].set_3d_properties(z)
                    lines[index_lines].set_markersize(1)
                    index_lines += 1
                count += 1


            writer.grab_frame()


    # create animation object
    ani = animation.FuncAnimation(fig, animate, frames=1, interval=0.5)

    # create PillowWriter object with higher fps value
    writer = PillowWriter(fps=1000)

    # save animation with PillowWriter
    ani.save('X_3DMotion.gif', writer=writer)









#----------------------- Animation ground Spots ------------------------- 
if animation_ground_spot == 1:
    
    fig, ax = plt.subplots()
    writer = PillowWriter(fps=10)

    # Define the ciruclar chief orbit of the Cubesat
    a_Cubesat = Re + 400000  # semi-major axis in km
    inclination = 90 #97.5 # determines how tilted the orbit is relative to the Earth's equator (deg)
    raan = 0
    argp = 0
    num_points = 3601
    T_Cubesat = 2*np.pi/w_Cubesat #period
    t_Cubesat = np.linspace(0, T_Cubesat, num_points)
    circular_orbit_Cubesat = circular_orbit(a_Cubesat, w_Cubesat, inclination, raan, argp, t_Cubesat)

    # Define the ciruclar orbit of the GNSS
    a_GNSS = Re + 23222000
    inclination_GNSS = 56 #97.5 # determines how tilted the orbit is relative to the Earth's equator (deg)
    raan_GNSS = 0
    argp_GNSS = 0
    T_GNSS = 2*np.pi/w_GNSS
    t_GNSS = np.linspace(0, T_GNSS, num_points)
    circular_orbit_GNSS = circular_orbit(a_GNSS, w_GNSS, inclination_GNSS, raan_GNSS, argp_GNSS, t_GNSS)


    x0_list = [9, 37, 78, 98, 107, 137, 158] #[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    phi0_list = [0.0, 90.0, 180.0, 270.0]
    images = []

    num_sat = len(phi0_list)*len(x0_list)+1  #min:20 / max:32
    print('Number of satellite :',num_sat)
    pos_sat = np.zeros((num_sat,3))

    center = np.array([0,0,0])

    t_obs = np.linspace(0, 3600, 10)
    def animate(i):

        ax.clear()

        # Plot chief satellite
        #s0, = ax.plot([], [], [], 'ko')
        dephasage = 0

        with writer.saving(fig, "Ground spot with shift2.gif", 100):

            for i in t_obs:

                i = int(i)

                Time = t_Cubesat[i]
                print("i =",i)
                print("Time =",Time)

                index=1
                pos_sat[0][0] = circular_orbit_Cubesat[0][0][i]
                pos_sat[0][1] = circular_orbit_Cubesat[0][1][i]
                pos_sat[0][2] = circular_orbit_Cubesat[0][2][i]

                # Plot Cubesats movement
                for x0 in x0_list:
                    for phi0 in phi0_list:
                        
                        x0_dot = 0.0
                        y0 = 2*x0_dot/w_Cubesat
                        y0_dot = -2*w_Cubesat*x0
                        z0 = x0
                        z0_dot = 0.0
                        #theta0 = -(90+phi0+dephasage)
                        if 600<i<1800:
                            theta0 = (90+phi0+dephasage)
                        else:
                            theta0 = -(90+phi0+dephasage)

                        x_rel, y_rel, z_rel = relative_3Dmotion(Time,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0+dephasage),np.radians(theta0),w_Cubesat)

                        
                        # Position du premier satellite
                        pos1 = np.array([circular_orbit_Cubesat[0][0][i], circular_orbit_Cubesat[0][1][i], circular_orbit_Cubesat[0][2][i]])
                        vit1 = np.array([circular_orbit_Cubesat[1][0][i], circular_orbit_Cubesat[1][1][i], circular_orbit_Cubesat[1][2][i]])

                        # Vecteur pointant du centre de la Terre vers le premier satellite
                        earth_to_sat1 = pos1/np.linalg.norm(pos1)
                        normal_dir = np.cross(pos1, vit1)
                        tangential_dir = np.cross(pos1, normal_dir)

                        # Normaliser les vecteurs
                        tangential_dir /= np.linalg.norm(tangential_dir)
                        normal_dir /= np.linalg.norm(normal_dir)

                        # Combiner les vecteurs pour obtenir la matrice de rotation
                        R = np.array([earth_to_sat1, tangential_dir, normal_dir]).T


                        orbit = np.vstack((x_rel, y_rel, z_rel))
                        x_rel = R.dot(orbit)[0]
                        y_rel = R.dot(orbit)[1]
                        z_rel = R.dot(orbit)[2]
                        
                        x = x_rel + circular_orbit_Cubesat[0][0][i]
                        y = y_rel + circular_orbit_Cubesat[0][1][i]
                        z = z_rel + circular_orbit_Cubesat[0][2][i]

                        pos_sat[index,:] = np.array([x[0], y[0], z[0]])
                        index+=1

                    dephasage += 90 / len(x0_list)

                # Plot specular point at each time
                pos_Cubesat = np.array([circular_orbit_Cubesat[0][0][i],circular_orbit_Cubesat[0][1][i],circular_orbit_Cubesat[0][2][i]])
                pos_GNSS = np.array([circular_orbit_GNSS[0][0][i],circular_orbit_GNSS[0][1][i],circular_orbit_GNSS[0][2][i]])
                
                intersections = sat_visibility(center, Re, pos_Cubesat, pos_GNSS)
                if intersections:
                    print("Satellites out of field of vision")

                else:
                    result = minimize_distance_on_sphere(pos_Cubesat, pos_GNSS, center, Re)
                    scale_factor = np.linalg.norm(center) + Re
                    specular_point = result.x * scale_factor
                
                # Plot ground spot
                width = 10000
                F, D, integral = Radiation_Pattern(pos_sat, pos_Cubesat, specular_point, 0.1)
                D = 10 * np.log10(D)
                print("D = {:.2f} dBi".format(np.amax(D)))

                G, F = ground_spot(specular_point, pos_Cubesat, pos_sat, width, integral)

                # Create a grid of coordinates
                x = np.linspace(-(width + 1) / 2, (width + 1) / 2, int(width / 10) + 1)
                y = np.linspace(-(width + 1) / 2, (width + 1) / 2, int(width / 10) + 1)
                X, Y = np.meshgrid(x, y)

                # Plot ground spot and add to images list
                images.append([ax.contourf(X, Y, G, 20)])
                print("image OK")
                #colorbar = plt.colorbar(images[-1], ax=ax)
                #colorbar.set_label("Intensity [dB]")
                ax.axis("square")

                writer.grab_frame()


    ax.set_xlabel("[m]")
    ax.set_ylabel("[m]")
    ax.set_title("Ground spot")

    # create animation object
    ani = animation.FuncAnimation(fig, animate, frames=10, interval=200, blit=False)

    # create PillowWriter object with higher fps value
    ani.save("animation.mp4", writer=writer)













#----------------------- Animation GNSS & Cubesats (specular point) ------------------------- 
if animation_specular_point == 1:
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the ciruclar chief orbit of the Cubesat
    a_Cubesat = 6371 + 4000  # semi-major axis in km
    inclination = 90 #97.5 # determines how tilted the orbit is relative to the Earth's equator (deg)
    raan = 0
    argp = 0
    num_points = 100
    T_Cubesat = 2*np.pi/w_Cubesat #period
    t_Cubesat = np.linspace(0, T_Cubesat, num_points)
    circular_orbit_Cubesat = circular_orbit(a_Cubesat, w_Cubesat, inclination, raan, argp, t_Cubesat)
    ax.plot(circular_orbit_Cubesat[0][0], circular_orbit_Cubesat[0][1],circular_orbit_Cubesat[0][2], color='g',label='Absolut Chief Orbit', markersize=1)

    # Define the ciruclar orbit of the GNSS
    a_GNSS = 6371 + 10000 #23222
    inclination_GNSS = 56 #97.5 # determines how tilted the orbit is relative to the Earth's equator (deg)
    raan_GNSS = 0
    argp_GNSS = 0
    T_GNSS = 50640 #Orbital period Galileo: 14h04
    w_GNSS = 2*np.pi/T_GNSS
    # T_GNSS = 2*np.pi/w_GNSS
    t_GNSS = np.linspace(0, T_GNSS, num_points)
    circular_orbit_GNSS = circular_orbit(a_GNSS, w_GNSS, inclination_GNSS, raan_GNSS, argp_GNSS, t_GNSS)
    ax.plot(circular_orbit_GNSS[0][0], circular_orbit_GNSS[0][1],circular_orbit_GNSS[0][2], color='orange',label='GNSS Orbit', markersize=1)

    x0_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    phi0_list = [0.0, 90.0, 180.0, 270.0]
    lines = []

    # Create the s values to set the data
    for x0 in x0_list:
        for phi0 in phi0_list:
            label = "s" + str(len(lines) + 1)
            line, = ax.plot([], [], [], 'bo', label=label)
            lines.append(line)

    # Valeurs supp pour le GNSS, le specular point et les droites d1 et d2
    colors =['bo','ro','r','r']
    for color in colors:
        label = "s" + str(len(lines) + 1)
        line, = ax.plot([], [], [], color, label=label)
        lines.append(line)

    print("Number of lines =",len(lines))




    def animate(i):
        # Plot chief satellite
        #s0, = ax.plot([], [], [], 'ko')
        dephasage = 0

        with writer.saving(fig, "GNSS and Cubesat motion2.gif", 100):

            for i, Time in enumerate(t_Cubesat):
                print(Time)
                index_lines = 0

                # Plot Cubesats movement
                for x0 in x0_list:
                    for phi0 in phi0_list:
                        
                        x0_dot = 0.0
                        y0 = 2*x0_dot/w_Cubesat
                        y0_dot = -2*w_Cubesat*x0
                        z0 = x0
                        z0_dot = 0.0
                        theta0 = -(90+phi0+dephasage)

                        x_rel, y_rel, z_rel = relative_3Dmotion(Time,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0+dephasage),np.radians(theta0),w_Cubesat)

                        
                        # Position du premier satellite
                        pos1 = np.array([circular_orbit_Cubesat[0][0][i], circular_orbit_Cubesat[0][1][i], circular_orbit_Cubesat[0][2][i]])
                        vit1 = np.array([circular_orbit_Cubesat[1][0][i], circular_orbit_Cubesat[1][1][i], circular_orbit_Cubesat[1][2][i]])

                        # Vecteur pointant du centre de la Terre vers le premier satellite
                        earth_to_sat1 = pos1/np.linalg.norm(pos1)
                        normal_dir = np.cross(pos1, vit1)
                        tangential_dir = np.cross(pos1, normal_dir)

                        # Normaliser les vecteurs
                        tangential_dir /= np.linalg.norm(tangential_dir)
                        normal_dir /= np.linalg.norm(normal_dir)

                        # Combiner les vecteurs pour obtenir la matrice de rotation
                        R = np.array([earth_to_sat1, tangential_dir, normal_dir]).T


                        orbit = np.vstack((x_rel, y_rel, z_rel))
                        x_rel = R.dot(orbit)[0]
                        y_rel = R.dot(orbit)[1]
                        z_rel = R.dot(orbit)[2]
                        
                        x = x_rel + circular_orbit_Cubesat[0][0][i]
                        y = y_rel + circular_orbit_Cubesat[0][1][i]
                        z = z_rel + circular_orbit_Cubesat[0][2][i]

                        lines[index_lines].set_data(x,y)
                        lines[index_lines].set_3d_properties(z)
                        lines[index_lines].set_markersize(1) # Plot des points plus petits
                        index_lines += 1
                    dephasage += 90 / len(x0_list)

                # Plot GNSS movement
                x_GNSS = circular_orbit_GNSS[0][0][i]
                y_GNSS = circular_orbit_GNSS[0][1][i]
                z_GNSS = circular_orbit_GNSS[0][2][i]

                lines[index_lines].set_data(x_GNSS,y_GNSS)
                lines[index_lines].set_3d_properties(z_GNSS)
                #lines[index_lines].set_markersize(1) 
                index_lines += 1


                # Plot specular point at each time
                pos_Cubesat = np.array([circular_orbit_Cubesat[0][0][i],circular_orbit_Cubesat[0][1][i],circular_orbit_Cubesat[0][2][i]])
                pos_GNSS = np.array([circular_orbit_GNSS[0][0][i],circular_orbit_GNSS[0][1][i],circular_orbit_GNSS[0][2][i]])
                
                intersections = sat_visibility(center, Re, pos_Cubesat, pos_GNSS)
                if intersections:

                    lines[index_lines].set_visible(False)
                    index_lines += 1
                    lines[index_lines].set_visible(False)
                    index_lines += 1
                    lines[index_lines].set_visible(False)
                    index_lines += 1
                    print("Satellites out of field of vision")

                else:
                    result = minimize_distance_on_sphere(pos_Cubesat, pos_GNSS, center, Re)
                    scale_factor = np.linalg.norm(center) + Re
                    specular_point = result.x * scale_factor

                    lines[index_lines].set_data(specular_point[0],specular_point[1])
                    lines[index_lines].set_3d_properties(specular_point[2])
                    #lines[index_lines].set_markersize(1)
                    lines[index_lines].set_visible(True)
                    index_lines += 1

                    lines[index_lines].set_data(np.array([specular_point[0], pos_Cubesat[0]]), np.array([specular_point[1], pos_Cubesat[1]]))
                    lines[index_lines].set_3d_properties(np.array([specular_point[2], pos_Cubesat[2]]))
                    lines[index_lines].set_linestyle('-')
                    lines[index_lines].set_visible(True)
                    index_lines += 1

                    lines[index_lines].set_data(np.array([specular_point[0], pos_GNSS[0]]), np.array([specular_point[1], pos_GNSS[1]]))
                    lines[index_lines].set_3d_properties(np.array([specular_point[2], pos_GNSS[2]]))
                    lines[index_lines].set_linestyle('-')
                    lines[index_lines].set_visible(True)

                #s0.set_data(0,0)
                #s0.set_3d_properties(0)

                writer.grab_frame()


    # -------------------
    # Si je veux supprimer le derriere et faire un fond noir
    ax.set_facecolor('black')

    ax.set_xlim(-12500,12500)
    ax.set_ylim(-12500,12500)
    ax.set_zlim(-11000,11000)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    # Cacher les lignes de grille
    ax.grid(False)
    # Cacher les ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Supprimer les bords des axes
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # -------------------

    # Earth Creation
    Re = 6371
    center = np.array([0,0,0])

    #texture_path = "/Users/eliotthubin/Downloads/earth_texture.jpg"
    #texture_image = Image.open(texture_path)
    #texture_image_rgba = texture_image.convert("RGBA")

    # Create a mesh grid for the sphere
    u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
    x = center[0] + Re * np.cos(u) * np.sin(v)
    y = center[1] + Re * np.sin(u) * np.sin(v)
    z = center[2] + Re * np.cos(v)
    ax.plot_surface(x, y, z, color='blue',alpha=0.2) #a enlever quand on veut plot la terre


    # Normalize texture coordinates to range [0, 1]
    #norm_u = (u - u.min()) / (u.max() - u.min())
    #norm_v = (v - v.min()) / (v.max() - v.min())
    #texture_image_resized = texture_image_rgba.resize((u.shape[0], v.shape[0]))
    #texture_data = np.array(texture_image_resized)/255.0  # Normalize RGBA values to [0, 1] range
    #alpha = 0.1  # Set the desired transparency value (0.0 - transparent, 1.0 - opaque)
    #texture_data[..., 3] = alpha
    #ax.plot_surface(x, y, z, facecolors=texture_data, rstride=1, cstride=1, shade=False) # Apply the texture to the surface of the sphere
    #---------------------


    # create animation object
    ani = animation.FuncAnimation(fig, animate, frames=1, interval=0.5)

    ax.view_init(azim=45, elev=45)

    # create PillowWriter object with higher fps value
    writer = PillowWriter(fps=1000,bitrate=5000)

    # save animation with PillowWriter
    ani.save('animation.gif', writer=writer)








"""
#----------------------- Animation absolute motion of Cubesats (x-y-z plan)---------------------- 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def animate(i):
    # Plot chief satellite
    #s0, = ax.plot([], [], [], 'ko')

    ax.set_xlim(-7000, 7000)
    ax.set_ylim(-7000, 7000)
    ax.set_zlim(-7000, 7000)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    #plt.title('Spiral formation')

    # Define the ciruclar chief orbit
    a = 6371 + 400  # semi-major axis in km
    inclination = 90 # determines how tilted the orbit is relative to the Earth's equator (deg)
    raan = 20
    argp = 30
    num_points = 100
    T = 2*np.pi/w_Cubesat #period
    t = np.linspace(0, T, num_points)
    circular_orbit_points = circular_orbit(a, w_Cubesat, inclination, raan, argp, t)
    ax.plot(circular_orbit_points[0][0], circular_orbit_points[0][1],circular_orbit_points[0][2], color='g',label='Absolut Chief Orbit', markersize=1)

    # Plot equatorial plane
    x, y = np.meshgrid(np.linspace(-7000, 7000, 100), np.linspace(-7000, 7000, 100))
    z = np.zeros_like(x)
    ax.plot_wireframe(x, y, z, color='b', alpha=0.1)


    x0_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    phi0_list = [0.0, 90.0, 180.0, 270.0]
    dephasage = 0
    lines = []


    # Create the s values to set the data
    for x0 in x0_list:
        for phi0 in phi0_list:
            label = "s" + str(len(lines) + 1)
            line, = ax.plot([], [], [], 'bo', label=label)
            lines.append(line)


    with writer.saving(fig, "SpiralAbsolut3DMotion1.gif", 100):

        for i, Time in enumerate(t):
            print(Time)
            index_lines = 0
            for x0 in x0_list:
                for phi0 in phi0_list:

                    x0_dot = 0.0
                    y0 = 2*x0_dot/w_Cubesat
                    y0_dot = -2*w_Cubesat*x0
                    z0 = 0
                    z0_dot = 0.0
                    theta0 = -(90+phi0+dephasage)

                    x_rel, y_rel, z_rel = relative_3Dmotion(Time,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0+dephasage),np.radians(theta0),w_Cubesat)

                    
                    # Position du premier satellite
                    pos1 = np.array([circular_orbit_points[0][0][i], circular_orbit_points[0][1][i], circular_orbit_points[0][2][i]])
                    vit1 = np.array([circular_orbit_points[1][0][i], circular_orbit_points[1][1][i], circular_orbit_points[1][2][i]])

                    # Vecteur pointant du centre de la Terre vers le premier satellite
                    earth_to_sat1 = pos1/np.linalg.norm(pos1)
                    normal_dir = np.cross(pos1, vit1)
                    tangential_dir = np.cross(pos1, normal_dir)

                    # Normaliser les vecteurs
                    tangential_dir /= np.linalg.norm(tangential_dir)
                    normal_dir /= np.linalg.norm(normal_dir)

                    # Combiner les vecteurs pour obtenir la matrice de rotation
                    R = np.array([earth_to_sat1, tangential_dir, normal_dir]).T


                    orbit = np.vstack((x_rel, y_rel, z_rel))
                    x_rel = R.dot(orbit)[0]
                    y_rel = R.dot(orbit)[1]
                    z_rel = R.dot(orbit)[2]
                    

                    x = x_rel + circular_orbit_points[0][0][i]
                    y = y_rel + circular_orbit_points[0][1][i]
                    z = z_rel + circular_orbit_points[0][2][i]

                    lines[index_lines].set_data(x,y)
                    lines[index_lines].set_3d_properties(z)
                    lines[index_lines].set_markersize(1)
                    index_lines += 1
                dephasage += 90 / len(x0_list)

            #s0.set_data(0,0)
            #s0.set_3d_properties(0)
            #print("vit1=",vit1)
            writer.grab_frame()


# create animation object
ani = animation.FuncAnimation(fig, animate, frames=1, interval=0.5)

# create PillowWriter object with higher fps value
writer = PillowWriter(fps=1000)

# save animation with PillowWriter
ani.save('animation.gif', writer=writer)
"""








"""
#----------------------- Animation phase shift formation (x-y-z plan)---------------------- 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot chief satellite
#s0, = ax.plot([], [], [], 'ko')

ax.set_xlim(-3000, 3000)
ax.set_ylim(-3000, 3000)
ax.set_zlim(-3000, 3000)


plt.xlabel('y [m]')
plt.ylabel('x [m]')
plt.title('Spiral formation')

metadata = dict(title='Movie', artist='codinglikemad')
writer = PillowWriter(fps=15, metadata=metadata)

num_points = 100
T = 2*np.pi/w #period
Time = np.linspace(0, T, num_points)

x0_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
phi0_list = [0.0, 90.0, 180.0, 270.0]
dephasage = 0
lines = []

# Create the s values to set the data
for x0 in x0_list:
    for phi0 in phi0_list:
        label = "s" + str(len(lines) + 1)
        line, = ax.plot([], [], [], 'bo', label=label)
        lines.append(line)


with writer.saving(fig, "SpiralRelative3DMotion.gif", 100):

    for t in Time:
        print(t)
        index_lines = 0
        for x0 in x0_list:
            for phi0 in phi0_list:

                x0_dot = 0.0
                y0 = 2*x0_dot/w
                y0_dot = -2*w*x0
                z0 = x0
                z0_dot = 0.0
                theta0 = -90-phi0-dephasage

                x, y, z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0+dephasage),np.radians(theta0),w)
                lines[index_lines].set_data(x, y)
                lines[index_lines].set_3d_properties(z)
                index_lines += 1
            dephasage += 90 / len(x0_list)

        #s0.set_data(0,0)
        #s0.set_3d_properties(0)
        
        writer.grab_frame()

#plt.show()
"""