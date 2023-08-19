import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


mu = 398600.5 #km3/s2
r_tgt = 6378.137 + 400 #km
w = np.sqrt(mu/r_tgt**3) #rad/s 
T = 2*np.pi/w/60
print('Period (min) =',T)
 

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

Spiral = 1
Y_fomation = 0



#----------------------- Animation Spiral formation (x-y plan)---------------------- 

if (Spiral):

    fig = plt.figure()

    # Plot chief satellite
    s0, = plt.plot([], [], 'ko')

    plt.xlim(-3000, 3000)
    plt.ylim(-3000, 3000)

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
            line, = plt.plot([], [], 'bo', label=label)
            lines.append(line)


    with writer.saving(fig, "SpiralRelativeMotion2.gif", 100):

        for t in Time:
            print(t)
            index_lines = 0
            for x0 in x0_list:
                for phi0 in phi0_list:

                    x0_dot = 0.5
                    y0 = 2*x0_dot/w
                    y0_dot = -2*w*x0
                    z0 = x0
                    z0_dot = 0.0
                    theta0 = 90+phi0+dephasage

                    x, y, z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0+dephasage),np.radians(theta0),w)
                    lines[index_lines].set_data(y, x)
                    index_lines += 1
                dephasage += 90 / len(x0_list)

            s0.set_data(0,0)
            
            writer.grab_frame()



        


#----------------------- Animation Spiral formation (y-z plan)---------------------- 

if (Y_fomation):
    fig  = plt.figure()
    s1, = plt.plot([], [], 'bo')
    s2, = plt.plot([], [], 'bo')
    s3, = plt.plot([], [], 'bo')
    s4, = plt.plot([], [], 'bo')
    s5, = plt.plot([], [], 'ko')
    s6, = plt.plot([], [], 'bo')
    s7, = plt.plot([], [], 'bo')


    plt.xlim(-3000, 3000)
    plt.ylim(-10, 50)

    plt.legend()
    plt.xlabel('z [m]')
    plt.ylabel('y [m]')
    plt.title('Relative motion in Y formationn')

    # Initialize the animation
    metadata = dict(title='Movie', artist='codinglikemad')
    writer = PillowWriter(fps=15, metadata=metadata)

    Time = np.linspace(0, 15000, 500)
    w = 2*np.pi/(60*T)

    with writer.saving(fig, "Y formation.gif", 100):

        for t in Time:
            x1,y1,z1 = relative_3Dmotion(t,0,0,10,0,0,0,np.radians(0),0,w)
            x2,y2,z2 = relative_3Dmotion(t,0,0,20,0,0,0,np.radians(0),0,w)
            x3,y3,z3 = relative_3Dmotion(t,0,0,30,0,100,1,np.radians(0),0,w)
            x4,y4,z4 = relative_3Dmotion(t,0,0,40,0,100,3,np.radians(0),0,w)
            x6,y6,z6 = relative_3Dmotion(t,0,0,25,0,100,0.5,np.radians(0),0,w)
            x7,y7,z7 = relative_3Dmotion(t,0,0,35,0,100,2,np.radians(0),0,w)

            s1.set_data(z1, y1)
            s2.set_data(z2, y2)
            s3.set_data(z3, y3)
            s4.set_data(z4, y4)
            s5.set_data(0,0)
            s6.set_data(-z6, y6)
            s7.set_data(-z7, y7)

            # Save the frame
            writer.grab_frame()
