import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def geo_to_cart_coord(longitude, latitude, altitude):

    longitude = np.radians(longitude)
    latitude = np.radians(latitude)

    x = altitude * np.cos(latitude) * np.cos(longitude)
    y = altitude * np.cos(latitude) * np.sin(longitude)
    z = altitude * np.sin(latitude)

    specular_cart_coord = np.array([x,y,z])

    return specular_cart_coord


# Compute if the Cubesat and the GNSS see each other by calculating the intersection points
# between the spherical Earth and the segment composed by the Cubesat and GNSS coordonates
def sat_visibility(center, radius, pos_Cubesat, pos_GNSS):
    """
    Calculate the intersection points between a sphere and a line segment in 3D space.
    
    Parameters:
    - center: numpy array representing the center of the sphere (x, y, z)
    - radius: radius of the sphere
    - pos_Cubesat: numpy array representing the coordinates of the first point on the line segment (x1, y1, z1)
    - pos_GNSS: numpy array representing the coordinates of the second point on the line segment (x2, y2, z2)
    
    Returns:
    - intersections: a list of intersection points (numpy arrays) or an empty list if no intersection
    """
    
    # Calculate the direction vector of the line segment
    vec_Cubesat_to_GNSS = pos_GNSS - pos_Cubesat
    
    # Calculate the vector from the segment's first point to the sphere center
    oc = pos_Cubesat - center
    
    # Calculate the coefficients of the quadratic equation
    a = np.dot(vec_Cubesat_to_GNSS, vec_Cubesat_to_GNSS)
    b = 2 * np.dot(vec_Cubesat_to_GNSS, oc)
    c = np.dot(oc, oc) - radius**2
    
    discriminant = b**2 - 4*a*c
    
    # No intersection
    if discriminant < 0:
        return []
    
    # Calculate the intersection points
    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    # Check if the intersection points are within the segment bounds
    intersections = []
    if 0 <= t1 <= 1:
        intersection1 = pos_Cubesat + t1 * vec_Cubesat_to_GNSS
        intersections.append(intersection1)
    if 0 <= t2 <= 1:
        intersection2 = pos_Cubesat + t2 * vec_Cubesat_to_GNSS
        intersections.append(intersection2)
    
    return intersections




# Function of minimization that compute the location of the specular point on the surface of a spherical Earth
def minimize_distance_on_sphere(pos_Cubesat, pos_GNSS, center, radius):
    """
    Trouve le point sur une sphère qui minimise la somme des distances entre pos_Cubesat, pos_GNSS et ce point sur la sphère.
    
    Parameters:
    - pos_Cubesat: numpy array représentant les coordonnées du premier point (x1, y1, z1) en mètres
    - pos_GNSS: numpy array représentant les coordonnées du deuxième point (x2, y2, z2) en mètres
    - center: numpy array représentant le centre de la sphère (x, y, z) en mètres
    - radius: rayon de la sphère en mètres
    
    Returns:
    - result: résultat de l'optimisation contenant le point sur la sphère (en coordonnées normalisées)
    """
    
    # Normalisation des coordonnées
    scale_factor = np.linalg.norm(center) + radius
    pos_Cubesat_normalized = pos_Cubesat / scale_factor
    pos_GNSS_normalized = pos_GNSS / scale_factor
    center_normalized = center / scale_factor
    radius_normalized = radius / scale_factor
    
    # Fonction objectif pour l'optimisation
    def objective_function(point):
        d1 = np.linalg.norm(point - pos_Cubesat_normalized)
        d2 = np.linalg.norm(point - pos_GNSS_normalized)
        return d1 + d2
    
    # Contrainte : la distance entre le point et la sphère doit être égale au rayon
    constraint = {'type': 'eq', 'fun': lambda point: np.linalg.norm(point - center_normalized) - radius_normalized}

    
    # Point initial pour l'optimisation
    initial_point = np.array([0.0, 0.0, 1.0])
    
    # Optimisation
    result = minimize(objective_function, initial_point, constraints=constraint)
    
    return result



# Function defines if the incident and reflection angle are the same (proof that the specular point computation works)
def compare_angles(d1, d2, normal):
    """
    Compare the angles of incidence and reflection.
    
    Parameters:
    - d1: numpy array representing the direction vector d1
    - d2: numpy array representing the direction vector d2
    - normal: numpy array representing the normal vector to the tangent plane
    
    Returns:
    - True if the angles of incidence and reflection are the same, False otherwise
    """

    # Normalize the direction vectors
    d1_normalized = d1 / np.linalg.norm(d1)
    d2_normalized = d2 / np.linalg.norm(d2)
    
    # Calculate the angles of incidence and reflection (for 3 decimals)
    angle_incidence = round(np.arccos(np.dot(d2_normalized, normal)),3)
    angle_reflection = round(np.arccos(np.dot(d1_normalized, normal)),3)

    print("angle_incidence=",np.degrees(angle_incidence))
    print("angle_reflection=",np.degrees(angle_reflection))
    
    # Compare the angles
    return np.isclose(angle_incidence, angle_reflection)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Data
center = np.array([0,0,0])
Re = 6371.0
pos_Cubesat = geo_to_cart_coord(100, 20, Re+1000)  #Cubesat coord
pos_GNSS = geo_to_cart_coord(0.0, 100, Re+10000)    #GNSS coord



print("---------------------------------------------------------------")
# ---------------------Compute visibility-------------------------
visibility = 0

intersections = sat_visibility(center, Re, pos_Cubesat, pos_GNSS)
if intersections:
    print("Intersection points:")
    for intersection in intersections:
        print(intersection)
else:
    print("No intersections, Cubesat and GNSS are in line-of-sight")
    visibility = 1

ax.scatter(pos_Cubesat[0], pos_Cubesat[1], pos_Cubesat[2], color='r', label='Cubesat')
ax.scatter(pos_GNSS[0], pos_GNSS[1], pos_GNSS[2], color='b', label='GNSS')
ax.plot([pos_Cubesat[0], pos_GNSS[0]], [pos_Cubesat[1], pos_GNSS[1]], [pos_Cubesat[2], pos_GNSS[2]], linestyle='dashed', color='orange')

ax.quiver(0, 0, 0, pos_Cubesat[0], pos_Cubesat[1], pos_Cubesat[2], color='r', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, pos_GNSS[0], pos_GNSS[1], pos_GNSS[2], color='b', arrow_length_ratio=0.1)

# Plot the intersection points
if intersections:
    for intersection in intersections:
        ax.scatter(intersection[0], intersection[1], intersection[2], color='g', marker='o')




# ---------------------Compute Minimisation---------------------
if visibility == 1:
    result = minimize_distance_on_sphere(pos_Cubesat, pos_GNSS, center, Re)
    scale_factor = np.linalg.norm(center) + Re
    specular_point = result.x * scale_factor
    #result = minimize_distance_on_sphere(pos_Cubesat, pos_GNSS, center, Re)
    #specular_point = result.x

    d1 = np.linalg.norm(specular_point - pos_Cubesat)
    d2 = np.linalg.norm(specular_point - pos_GNSS)

    ax.scatter(specular_point[0], specular_point[1], specular_point[2], color='k', label='specular_point')
    print("Specular Point :", specular_point)
    print("Distance d1 (pos_Cubesat -> Specular Point) :", d1)
    print("Distance d2 (pos_GNSS -> Specular Point) :", d2)
    print("Somme des distances (d1 + d2) :", d1 + d2)




# ---------------------Compare incidence and reflection angles---------------------
if visibility == 1:
    d1_vec = np.array(pos_Cubesat - specular_point)
    d2_vec = np.array(pos_GNSS - specular_point)
    normal = (specular_point - center)
    normal_normalized = (specular_point - center)/np.linalg.norm(specular_point - center)

    ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], color='k')
    ax.quiver(specular_point[0], specular_point[1], specular_point[2], d1_vec[0], d1_vec[1], d1_vec[2], color='k', arrow_length_ratio=0.1)
    ax.quiver(specular_point[0], specular_point[1], specular_point[2], d2_vec[0], d2_vec[1], d2_vec[2], color='k', arrow_length_ratio=0.1)

    result = compare_angles(d1_vec, d2_vec, normal_normalized)
    if result == True:
        print("Incidence and reflection angles are equal")
    else:
        print("Incidence and reflection angles are not equal")
print("---------------------------------------------------------------")


# Plot the sphere
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
x = center[0] + Re * np.cos(u) * np.sin(v)
y = center[1] + Re * np.sin(u) * np.sin(v)
z = center[2] + Re * np.cos(v)
ax.plot_surface(x, y, z, color='b', alpha=0.5)

ax.set_xlim(-12500,12500)
ax.set_ylim(-12500,12500)
ax.set_zlim(-11000,11000)

ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')

plt.show()












