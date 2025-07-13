'''
This Python code processes a series of images to estimate the 3D dimensions of a pipe using stereo vision, calculates the pipe's radius, transforms points to world coordinates, and then visualizes the results by drawing an ideal cylinder. Here's a breakdown of how each part of the code works:

1. Loading Camera Parameters and Setup:

Loads camera matrix and distortion coefficients from .npy files, essential for correcting lens distortions.
Defines camera resolution, sensor size, laser position, and various points used for calibration and transformation.
Sets up parameters like focal length, step value (for incremental processing), ideal radius of the pipe, and the path to the image folder.
2. Plane Equation Calculation:

find_plane_equation: Calculates the equation of the laser plane using three points on it. This plane equation is crucial for 3D point reconstruction.
3. Image Loading and Preprocessing:

Loads images from the specified folder.
thresh: This function preprocesses each image to extract the laser line:
Corrects lens distortion.
Applies a region of interest (ROI) and darkens the rest of the image.
Converts the ROI to HSV color space and applies a mask to isolate the laser light.
Converts to grayscale, enhances contrast using CLAHE, and applies Gaussian blur.
Performs binary thresholding and morphological operations (erosion, skeletonization) to get a thin laser line.
Extracts the pixel coordinates of the laser line.
4. 3D Point Calculation:

PipeDirection: Calculates the direction vector of the pipe.
calculate3dPoints: Converts 2D pixel coordinates to 3D world coordinates:
Calculates a unit vector from the camera center to each pixel.
Calculates the 3D point on the laser plane using the plane equation.
Shifts the 3D points along the pipe's direction vector based on the step value.
5. Arc Finding and Radius Calculation:

findArcs: Processes the 3D points to find arcs and calculate the pipe's radius:
Filters points to remove outliers.
Identifies points lying on the laser plane.
Calculates the center and radius of each arc using a least-squares fitting method.
Calculates the mean radius and the mean of the squared differences between the calculated radii and the ideal radius.
6. Coordinate Transformation:

Transformation: Transforms 3D points from the camera's local coordinate system to the world coordinate system. This involves calculating rotation matrices based on known points in both coordinate systems.
7. Visualization:

plotshape: Visualizes the 3D points and an ideal cylinder:
Uses plotly to create 3D and 2D scatter plots of the transformed 3D points.
Generates points for an ideal cylinder based on the calculated center, radius, and direction.
Applies a rotation matrix to align the cylinder with the pipe's direction.
Adds the cylinder surface to the 3D plot.
Dynamically sets the 3d graph's axis ranges to best fit the plotted data.
Enforces equal scaling for the 2d graphs.
Displays the plots.
'''



import cv2
import numpy as np
from skimage.morphology import skeletonize
import os
import json
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import the calibrated transportation parameters.
with open(r"RELATIVE_PATH\Matrixparameters.json", "r") as file: 
    data = json.load(file)

optimized_R = np.array(data["optimized_R"])
optimized_t = np.array(data["optimized_t"])

folder_path = r"RELATIVE_PATH"
camera_matrix = np.load(r"RELATIVE_PATH\cameraMatric.npy")
distortion_coeffs = np.load(r"RELATIVE_PATH\distCoeffs.npy") 

Stepvalue = 0.250 
ideal_radius = 36

resolution_x = 1280  
Xsensor_size = 3.58 
resolution_y = 720  
Ysensor_size = 2.02 
focal_length = 4  

laser_position = np.array([-221.458, 0.000, 29.306])  
point_on_conveyor_under_camera = np.array([0.000, 0.000, 318.436])  
point_on_top_pipe_other = np.array([-29.709, 20.000, 279.649])  
point_on_top_pipe = np.array([-29.709, 0.000, 279.649])  
point_on_Pipe_direction = np.array([-109.097, 0.000, 340.455]) 
sensor_center = np.array([0.000, 0.000, 0.000])  

P1_local = point_on_top_pipe
P1_world = np.array([-0.000,0, 26.000])
P2_local = point_on_Pipe_direction 
P2_world = np.array([-100.000,  0, 26.000])
center_local = np.array([-13.899, 0,300.290])

image_names = []
images_path = []
images=[] 


def find_plane_equation(point1, point2, point3):
    v1 = np.subtract(point2, point1)  
    v2 = np.subtract(point3, point1)  
    normal = np.cross(v1, v2)  
    d = -np.dot(normal, point1)  
    return normal, d
    
normal, d = find_plane_equation(laser_position, point_on_conveyor_under_camera, point_on_top_pipe_other)
print("Laser plane's equation calculated.")

print("Loading images from folder...")

# Initial code to load images from a folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG')):  
        image_names.append(filename)
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        images_path.append(image_path)
        images.append(image)
         
print(f"{len(images)} images loaded.")

def thresh(img):
    """
    Processes an input image to extract the skeleton of specific features within a region of interest (ROI).
    
    This function undistorts the image, defines an ROI, darkens the area outside it, applies color filtering in HSV space,
    and performs a series of image processing steps (grayscale conversion, blurring, thresholding, sharpening, contrast
    enhancement, erosion, normalization, and skeletonization) to produce a binary skeleton image and its pixel coordinates.
    
    Args:
        img (numpy.ndarray): The input image in BGR format.
    
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Transposed array of coordinates (x, y) of skeleton pixels.
            - numpy.ndarray: The skeleton image as an 8-bit unsigned integer array.
    
    Notes:
        - Requires global variables `camera_matrix` and `distortion_coeffs` for undistortion.
    """
    undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
    x, y, w, h = 450, 200, 250, 350
    mask = np.zeros_like(undistorted_img)
    mask[y:y+h, x:x+w] = 255

    darkened_image = np.copy(undistorted_img).astype(float)
    darkened_image *= 0.15
    darkened_image = darkened_image.astype(np.uint8)
    crop_image = np.where(mask == 255, undistorted_img, darkened_image)

    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([110,0,190])
    upper_bound = np.array([255,170,255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    filtered_image = cv2.bitwise_and(crop_image, crop_image, mask=mask)

    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 105)
    _, binary_image = cv2.threshold(blur, 55, 255, cv2.THRESH_BINARY)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    filtered_binary = cv2.filter2D(binary_image, -1, kernel)
 
    clipLimit = 15.0
    tileGridSize = (3, 3)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    enhanced_gray = clahe.apply(filtered_binary)
    eroded_binary = cv2.erode(enhanced_gray, np.ones((1, 1), np.uint8), iterations=1) 
    eroded_binary_normalized = cv2.normalize(eroded_binary, None, 0, 255, cv2.NORM_MINMAX)
    binary_input = (eroded_binary_normalized // 255).astype(bool)
    skeleton = skeletonize(binary_input)
    skeleton_image = (skeleton * 255).astype(np.uint8)
    right_edges = np.where(skeleton > 0)

    return np.array(right_edges).T, skeleton_image

# Processing loop using thresh
pointslist = []
skelimages = []

print(f"Applying filters on images...") 

for i in range(len(images)): 
    threshimage, skimage = thresh(images[i])
    if 250 < len(threshimage) < 320: #filter images with too much points.(probably there were extra laser line)
        pointslist.append(threshimage) 
        skelimages.append(skimage)
        
print(f"skeleton points added to the list from {len(pointslist)} of images.")

def PipeDirection(point_on_top_pipe, point_on_Pipe_direction):
    """
    Computes the unit vector representing the direction from one point to another, typically used to define pipe direction.
    """
    Pv1 = point_on_top_pipe 
    Pv2 = point_on_Pipe_direction
    vector = Pv2 - Pv1
    magnitude = np.linalg.norm(vector)
    unit_vector_pipe = vector / magnitude  
    return unit_vector_pipe  
    
calculated_R = []

def Transformation(P_test_local, P1_local=P1_local, P1_world=P1_world, P2_local=P2_local, P2_world=P2_world, center_local=center_local):
    """
    Computes a transformation matrix from local to world coordinates and applies it to a given point.
    
    This function defines local and world coordinate bases using two reference points, calculates the rotation matrix
    to align these bases, and transforms the input point from local to world coordinates. The rotation matrix is also
    stored in a global list `calculated_R`.
    
    Args:
        P_test_local (numpy.ndarray): The 3D point in local coordinates to transform.
        P1_local (numpy.ndarray): First reference point in local coordinates (default defined globally).
        P1_world (numpy.ndarray): First reference point in world coordinates (default defined globally).
        P2_local (numpy.ndarray): Second reference point in local coordinates (default defined globally).
        P2_world (numpy.ndarray): Second reference point in world coordinates (default defined globally).
        center_local (numpy.ndarray): Center of the local coordinate system (default defined globally).
    
    Returns:
        numpy.ndarray: The transformed 3D point in world coordinates.
    
    Notes:
        - Modifies the global list `calculated_R` by appending the computed rotation matrix.
    """
    v1_local = P1_local - center_local
    v2_local = P2_local - center_local
    v1_world = P1_world.copy()
    v2_world = P2_world.copy()

    e1_local = v2_local / np.linalg.norm(v2_local)
    proj_v1_on_e1 = np.dot(v1_local, e1_local) * e1_local
    v1_local_ortho = v1_local - proj_v1_on_e1
    e2_local = v1_local_ortho / np.linalg.norm(v1_local_ortho)
    e3_local = np.cross(e1_local, e2_local)

    L = np.column_stack((e1_local, e2_local, e3_local))
    e1_world = v2_world / np.linalg.norm(v2_world)
    proj_v1_on_e1_world = np.dot(v1_world, e1_world) * e1_world
    v1_world_ortho = v1_world - proj_v1_on_e1_world
    e2_world = v1_world_ortho / np.linalg.norm(v1_world_ortho)
    e3_world = np.cross(e1_world, e2_world)
    W = np.column_stack((e1_world, e2_world, e3_world))
    R = W @ L.T

    calculated_R.append(R)
    def local_to_world(P_local):
        return R @ (P_local - center_local)
        
    P_world = local_to_world(P_test_local)

    return P_world

def optimized_transformation(P_local, center_local, R_opt, t_opt):
    """
    Applies an optimized transformation to a point from local to world coordinates using precomputed rotation and translation.
    """
    return R_opt @ (P_local - center_local) + t_opt

def calculate3dPoints(resolution_x, Xsensor_size, resolution_y, Ysensor_size, focal_length, Stepvalue, pointslist):
    """
    This function converts 2D pixel coordinates to 3D points in local coordinates using camera intrinsics, shifts them along
    a pipe direction, and transforms them to world coordinates using both standard and optimized transformations. It uses
    helper functions to normalize vectors, compute ray directions, and intersect rays with a plane.
    
    Args:
        resolution_x (int): Horizontal resolution of the image in pixels.
        Xsensor_size (float): Horizontal size of the camera sensor in physical units (e.g., mm).
        resolution_y (int): Vertical resolution of the image in pixels.
        Ysensor_size (float): Vertical size of the camera sensor in physical units (e.g., mm).
        focal_length (float): Focal length of the camera in physical units (e.g., mm).
        Stepvalue (float): Step size for shifting points along the pipe direction.
        pointslist (list): List of 2D point arrays (each array contains (y, x) coordinates from skeleton images).
    
    Returns:
        tuple: A tuple containing:
            - list: 3D points shifted along the pipe direction.
            - list: 3D points in local coordinates for each image.
            - list: 3D points transformed to world coordinates using `Transformation`.
            - list: 3D points transformed to world coordinates using `optimized_transformation`.
    
    Notes:
        - Relies on global variables: `normal`, `point_on_conveyor_under_camera`, `d`, `P1_local`, `P1_world`, `P2_local`,
          `P2_world`, `center_local`, `optimized_R`, `optimized_t`, `point_on_top_pipe`, `point_on_Pipe_direction`.
    """
    def normalize(vector):
        norm = np.linalg.norm(vector)
        if norm == 0: 
            return vector
        return vector / norm
    
    def calculate_U(xs, ys, Xsensor_size, Ysensor_size, focal_length): 
        x = xs - (Xsensor_size / 2)
        y = (ys - (Ysensor_size / 2))  
        u = np.array([x, y, focal_length])  
        u = normalize(u)  
        return u 
    
    def calculate_z(u, normal, Ll):
        u1, u2, u3 = u  
        k1, k2, k3 = normal  
        k4 = Ll[2]
        lambda_value = (k3 * k4) / ((k1 * u1) + (k2 * u2) + (k3 * u3))  
        lambda_value = lambda_value 

        x = lambda_value * u1  
        y = lambda_value * u2  
        z = lambda_value * u3  
        return x, y, z
            
    unit_vector_pipe = PipeDirection(point_on_top_pipe, point_on_Pipe_direction)
    Images3d = []
    pointslistfor_Arc = []
    pointslistfor_Arc_world = []
    pointslistfor_Arc_world_optimized = []
    
    for i in range(len(pointslist)):
        Landa = Stepvalue * (i)
        Image3dworld = []
        Image3dworldoptimized = []
        Image3d = []
        
        for point_id in range(len(pointslist[i])):
            point = pointslist[i][point_id]
            yp, xp = point
            xp_insensor = (xp * Xsensor_size) / resolution_x
            yp_insensor = (yp * Ysensor_size) / resolution_y
            u = calculate_U(xp_insensor, yp_insensor, Xsensor_size, Ysensor_size, focal_length)    
            x, y, z = calculate_z(u, normal, point_on_conveyor_under_camera)
            point3d = x, y, z
            Image3d.append(point3d)

            if abs((normal[0] * point3d[0]) + (normal[1] * point3d[1]) + (normal[2] * point3d[2]) + d) <= 0:
                 point3dworld = Transformation(point3d, P1_local, P1_world, P2_local, P2_world, center_local)
                 point3dworldoptimized = optimized_transformation(point3d, center_local, optimized_R, optimized_t)
                 Image3dworld.append(point3dworld)
                 Image3dworldoptimized.append(point3dworldoptimized)
            
            point3d_shifted = (x + Landa * unit_vector_pipe[0], y + Landa * unit_vector_pipe[1], z + Landa * unit_vector_pipe[2])
            Images3d.append(point3d_shifted)  

        pointslistfor_Arc.append(Image3d)
        pointslistfor_Arc_world.append(Image3dworld)
        pointslistfor_Arc_world_optimized.append(Image3dworldoptimized)
        
    return Images3d, pointslistfor_Arc, pointslistfor_Arc_world, pointslistfor_Arc_world_optimized

Points3d, pointslistfor_Arc, pointslistfor_Arc_world, pointslistfor_Arc_world_optimized = calculate3dPoints(resolution_x, Xsensor_size, resolution_y, Ysensor_size, focal_length, Stepvalue, pointslist)

list3d_world = []

for point in Points3d :
    pointworld = optimized_transformation(point, center_local, optimized_R, optimized_t)
    list3d_world.append(pointworld)


def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.
    
    Parameters:
        point1 (tuple/list): First point (x1, y1, z1)
        point2 (tuple/list): Second point (x2, y2, z2)
        
    Returns:
        float: Distance between the points
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]
    squared_distance = dx**2 + dy**2 + dz**2
    return math.sqrt(squared_distance)



def findArcs(Arc, normal_world, d_world):
    """
    Processes a list of 3D point sets (arcs) to fit circular arcs, compute their centers and radii,
    and shift points along a pipe direction for further analysis.

    This function iterates over each set of points in `Arc`, filters them based on differences in y and z coordinates,
    computes the average y and z coordinates, fits a circular arc to the points while preferring a center near specified
    y and z values, and shifts the points along a pipe direction using a unit vector. It handles cases where no points
    are present or when arc fitting fails, appending valid results to global lists.

    Args:
        Arc (list): A list of arrays, where each array contains 3D points (x, y, z) representing an arc.
        normal_world (numpy.ndarray): The normal vector of the plane in world coordinates [a, b, c].
        d_world (float): The plane constant in the equation a*x + b*y + c*z + d = 0.

    Returns:
        None: This function does not return any value but appends results to global lists `arcpipes` and `Centers`.

    Notes:
        - Modifies global lists `arcpipes`, `Centers`, and potentially `radiusMeans` (logic incomplete in original code).
        - Relies on global variables: `Stepvalue`, `point_on_top_pipe`, `point_on_Pipe_direction`, `center_local`, 
          `optimized_R`, `optimized_t`.
        - Uses external functions: `calculate_distance`, `PipeDirection`, `optimized_transformation`, `hyper_fit`.
        - Skips processing if no points are present or if arc fitting results in an invalid center (e.g., |y_center| > 10).
    """
    number = 0
    for points in Arc:
        if points:
            pass
        else:
            print(f"there is no any point")
            continue

        # cv2.imshow(f"Skeleton ", skelimages[number])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  
          
        y_arc = []
        Z_arc = []
        distance = []
        
        def filter_points(points):
            """
            Filters points based on differences in y and z coordinates to identify unique segments.

            This nested function processes the points to detect significant changes in y and z coordinates,
            which are used to identify distinct parts of the arc. It appends points to `y_arc` and `Z_arc`
            when a unique condition is met based on a threshold.

            Args:
                points (numpy.ndarray): Array of 3D points (x, y, z).

            Returns:
                numpy.ndarray: Filtered array of points.
            """
            filtered_points = []
            unique = False
            for i in range(len(points)):
                if i > 1:
                    j = i - 1
                    difZ = abs(points[i][2] - points[j][2])
                    difY = abs(points[i][1] - points[j][1])
                
                    if difZ * difY > 75:
                        # print(f"X1:{round(points[j][0],3)} Y1:{round(points[j][1],3)} Z1:{round(points[j][2],3)}")
                        # print(f"X2:{round(points[i][0],3)} Y2:{round(points[i][1],3)} Z2:{round(points[i][2],3)}")
                        distance.append(points[j])
                        distance.append(points[i])
                        unique = not unique
                    if unique:
                        filtered_points.append(points[i])
                        y_arc.append(points[i][1])
                        Z_arc.append(points[i][2])
                
            return np.array(filtered_points)

        Arc_points = filter_points(points)

        if len(distance) < 3: # check if the line upper and down of the arc are visible
            # for i in range(len(points)):
            #    print(f"X:{round(points[i][0],3)} Y:{round(points[i][1],3)} Z:{round(points[i][2],3)}")
            print(f"Warning: image number {number} was not completed")
            number = number + 1
            # wait = input("waiting")
            continue
        # the the distance of seperated laser line on the surface from each other
        # print(f"distance is {calculate_distance(distance[0], distance[3])} number: {number}")
        
        x = [point[0] for point in Arc_points]
        y = [point[1] for point in Arc_points]
        z = [point[2] for point in Arc_points]
        Landa = Stepvalue * number
        
        unit_vector_pipe = PipeDirection(
            optimized_transformation(point_on_top_pipe, center_local, optimized_R, optimized_t),
            optimized_transformation(point_on_Pipe_direction, center_local, optimized_R, optimized_t)
        )
        
        pointsArc_shifted = (
            x + (Landa * unit_vector_pipe[0]),
            y + (Landa * unit_vector_pipe[1]),
            z + (Landa * unit_vector_pipe[2])
        )
        
        for i in range(len(pointsArc_shifted[0])):
            point = (pointsArc_shifted[0][i], pointsArc_shifted[1][i], pointsArc_shifted[2][i])
            arcpipes.append(point)

        averageY = sum(y_arc) / len(y_arc)
        averageZ = sum(Z_arc) / len(Z_arc)
        closest_numberY = min(y_arc, key=lambda x: abs(x - averageY))
        closest_numberZ = min(Z_arc, key=lambda x: abs(x - averageZ))
        y_center = closest_numberY
        Z_center = closest_numberZ
        
        # from circle_fit import hyper_fit  

        def find_radius_center(preferred_y, preferred_z, normal_vector, d_vector, x, y, z, regularization_weight=1.0):
            """
            Fits a circular arc to 3D points, with the center on a given plane and close to preferred y and z coordinates.

            This nested function projects points onto a plane defined by a normal vector and constant, applies a 
            regularization to prefer centers near (preferred_y, preferred_z), and fits a circle using least squares 
            with regularization. It handles edge cases like zero normal vectors and computes the 3D center and radius.

            Args:
                preferred_y (float): Preferred y-coordinate of the center.
                preferred_z (float): Preferred z-coordinate of the center.
                normal_vector (numpy.ndarray): Plane normal vector [a, b, c].
                d_vector (float): Plane constant in a*x + b*y + c*z + d = 0.
                x (list): List of x coordinates of points.
                y (list): List of y coordinates of points.
                z (list): List of z coordinates of points.
                regularization_weight (float, optional): Weight for preferring the center near (preferred_y, preferred_z). 
                    Defaults to 1.0.

            Returns:
                tuple: (center, radius)
                    - center (list): [x_c, y_c, z_c] coordinates of the center.
                    - radius (float): Radius of the fitted circle.
            """
            normal = np.array(normal_vector)
            points = np.column_stack((x, y, z))
                        
            norm_magnitude = np.linalg.norm(normal)
            if norm_magnitude == 0:
                raise ValueError("Normal vector cannot be zero.")
            normal = normal / norm_magnitude
            d = d_vector / norm_magnitude
            a, b, c = normal
            
            t = (points @ normal + d)
            projected_points = points - np.outer(t, normal)
            Q = projected_points[0]

            if abs(a) >= max(abs(b), abs(c)):
                U = np.array([0, -c, b])
            elif abs(b) >= max(abs(a), abs(c)):
                U = np.array([c, 0, -a])
            else:
                U = np.array([-b, a, 0])
            U = U / np.linalg.norm(U)
            V = np.cross(normal, U)  
            
            points_2d = np.column_stack([
                (projected_points - Q) @ U,
                (projected_points - Q) @ V
            ])
            
            if a != 0:
                x_pref = -(b * preferred_y + c * preferred_z + d) / a
            else:
                x_pref = 0  
            C_pref = np.array([x_pref, preferred_y, preferred_z])
            t_pref = (C_pref @ normal + d)
            C_pref_proj = C_pref - t_pref * normal
            C_pref_2d = np.array([(C_pref_proj - Q) @ U, (C_pref_proj - Q) @ V])
            
            A = np.column_stack((points_2d[:, 0], points_2d[:, 1], np.ones(len(points_2d))))
            d_vec = -(points_2d[:, 0]**2 + points_2d[:, 1]**2)
            
            w = regularization_weight
            A_aug = np.vstack([
                A,
                [w, 0, 0],           
                [0, w, 0]            
            ])
            d_aug = np.concatenate([
                d_vec,
                [w * (-2 * C_pref_2d[0]), w * (-2 * C_pref_2d[1])]
            ])
            
            coeffs, _, _, _ = np.linalg.lstsq(A_aug, d_aug, rcond=None)
            a, b, c = coeffs
            xc_reg = -a / 2
            yc_reg = -b / 2
            r_reg = np.sqrt((a/2)**2 + (b/2)**2 - c)
            
            center_3d = Q + xc_reg * U + yc_reg * V
            center = [float(center_3d[0]), float(center_3d[1]), float(center_3d[2])]
            radius = float(r_reg)

            if abs(preferred_y) > 10:
                print("Error in finding the Arc....!!")
                print(f"image number is {number}")
                print(f"y_center:{y_center} Z_center:{Z_center}")
                print(f"radius is {radius}")
                print("Skip the image")
            else:
                Centers.append(center)
                radiusMeans.append(radius)
            return center, radius

        # Call the nested function to find center and radius
        find_radius_center(y_center, Z_center, normal_world, d_world, x, y, z, regularization_weight=1.0)
        number = number + 1 

laser_position_world_optimized = optimized_transformation(laser_position, center_local, optimized_R, optimized_t)
point_on_conveyor_under_camera_world_optimized = optimized_transformation(point_on_conveyor_under_camera, center_local, optimized_R, optimized_t)
point_on_top_pipe_other_world_optimized = optimized_transformation(point_on_top_pipe_other, center_local, optimized_R, optimized_t)

normal_world_optimized, d_world_optimized = find_plane_equation(laser_position_world_optimized, point_on_conveyor_under_camera_world_optimized, point_on_top_pipe_other_world_optimized)
arcpipes = []
radiusMeans = []
Centers = []

findArcs(pointslistfor_Arc_world_optimized, normal_world_optimized, d_world_optimized)
centers_array = np.array(Centers)  
mean_centers = np.mean(centers_array, axis=0)
filtered_radius = [num for num in radiusMeans if 25 <= num < 26.5] 
print(f"optimized: len all:{len(radiusMeans)} len filter : {len(filtered_radius)}")
print(f"optimized: Mean center across all fits: {round(mean_centers[0],6)} {round(mean_centers[1],3)} {round(mean_centers[2],3)}")
print(f"optimized: max radius is: {round(np.max(filtered_radius),6)}")
print(f"optimized: mean radius is: {round(np.mean(filtered_radius),6)}")
print(f"optimized: median radius is: {round(np.median(filtered_radius),6)}")
print(f"optimized: min radius is: {round(np.min(filtered_radius),6)}")
print(f"optimized: radius tolerance is: {round(np.max(filtered_radius) - np.min(filtered_radius),6)}")

x = [point[0] for point in arcpipes]
y = [point[1] for point in arcpipes]
z = [point[2] for point in arcpipes]

def plotshape(x, y, z, ideal_radius, center, unit_vec, min_radius, max_radius):
    """
    Plots 3D points and two cylinders with different radii.

    Args:
        x, y, z: Coordinates of the 3D points.
        ideal_radius: Radius of the first cylinder.
        center: Center point of the cylinders.
        unit_vec: Direction vector of the cylinders.
        other_radius: Radius of the second cylinder (optional).
    """

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]],
        subplot_titles=('3D View', 'Top View (XY)', 'Front View (XZ)', 'Side View (YZ)')
    )

    
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=1, color=z, colorscale='Inferno', colorbar=dict(title='Z Value')),
        ),
        row=1, col=1
    )
    

    fig.add_trace(
        go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode='markers',
            marker=dict(size=5, color='green'),  
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[center[0]], y=[center[1]],
            mode='markers',
            marker=dict(size=5, color='green'),
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[center[0]], y=[center[2]],
            mode='markers',
            marker=dict(size=5, color='green'),
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[center[1]], y=[center[2]],
            mode='markers',
            marker=dict(size=5, color='green'),
        ),
        row=2, col=2
    )

    
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=1, color=z, colorscale='Inferno'),
        ),
        row=1, col=2
    )


    
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=1, color=z, colorscale='Inferno'),
        ),
        row=1, col=2
    )

    
    fig.add_trace(
        go.Scatter(
            x=x, y=z,
            mode='markers',
            marker=dict(size=1, color=z, colorscale='Inferno'),
        ),
        row=2, col=1
    )

    
    fig.add_trace(
        go.Scatter(
            x=y, y=z,
            mode='markers',
            marker=dict(size=1, color=z, colorscale='Inferno'),
        ),
        row=2, col=2
    )

    def draw_cylinder(radius, color='Viridis'):
        nonlocal x_global, y_global, z_global 
        r = radius
        height = 500
        num_theta = 50
        num_z = 50

        theta = np.linspace(0, 2 * np.pi, num_theta)
        z_local = np.linspace(-height / 2, height / 2, num_z)
        theta, z_local = np.meshgrid(theta, z_local)
        x_local = r * np.cos(theta)
        y_local = r * np.sin(theta)

        def rotation_matrix_from_vectors(a, b):
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            if s < 1e-8:
                return np.eye(3)
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
            return R

        R = rotation_matrix_from_vectors(np.array([0, 0, 1]), unit_vec)

        pts_local = np.array([x_local, y_local, z_local])
        pts_local_flat = pts_local.reshape(3, -1)
        pts_rotated = R @ pts_local_flat
        pts_rotated = pts_rotated.reshape(3, num_z, num_theta)
        x_global = pts_rotated[0] + center[0]
        y_global = pts_rotated[1] + center[1]
        z_global = pts_rotated[2] + center[2]

        fig.add_trace(
            go.Surface(
                x=x_global, y=y_global, z=z_global,
                opacity=0.5,
                colorscale=color,
                showscale=False
            ),
            row=1, col=1
        )

    x_global, y_global, z_global = np.array([]), np.array([]), np.array([]) 
    draw_cylinder(ideal_radius)  
    
    draw_cylinder(min_radius, color='Plasma') 
    draw_cylinder(max_radius, color='tealgrn') 

    
    all_x = np.concatenate([np.array(x), x_global.flatten()])
    all_y = np.concatenate([np.array(y), y_global.flatten()])
    all_z = np.concatenate([np.array(z), z_global.flatten()])

    
    x_mid = (all_x.min() + all_x.max()) / 2
    y_mid = (all_y.min() + all_y.max()) / 2
    z_mid = (all_z.min() + all_z.max()) / 2

    
    max_half_range = max((all_x.max() - all_x.min()) / 2,
                         (all_y.max() - all_y.min()) / 2,
                         (all_z.max() - all_z.min()) / 2)

    
    x_range = [x_mid - max_half_range, x_mid + max_half_range]
    y_range = [y_mid - max_half_range, y_mid + max_half_range]
    z_range = [z_mid - max_half_range, z_mid + max_half_range]

    
    fig.update_layout(
        width=1800, height=1000,
        scene=dict(
            xaxis=dict(title='X Label', range=x_range),
            yaxis=dict(title='Y Label', range=y_range),
            zaxis=dict(title='Z Label', range=z_range),
            aspectmode='cube',
            camera=dict(
                projection=dict(type='orthographic')
            )
        ),
        xaxis2=dict(title='X Label'),
        yaxis2=dict(title='Y Label'),
        xaxis3=dict(title='X Label'),
        yaxis3=dict(title='Z Label'),
        xaxis4=dict(title='Y Label'),
        yaxis4=dict(title='Z Label'),
    )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=2)
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=2, col=1)
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=2, col=2)

    fig.show()


unit_vector_pipe_world = PipeDirection(optimized_transformation(point_on_top_pipe, center_local, optimized_R, optimized_t),
                             optimized_transformation(point_on_Pipe_direction, center_local, optimized_R, optimized_t))

waitiing = input(f"press Enter to plot the shape")
plotshape(x, y, z, ideal_radius, mean_centers, unit_vector_pipe_world, np.min(radiusMeans), np.max(radiusMeans) )




