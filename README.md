3D Pipe Scanner using Laser Triangulation
[https://github.com/Arashpirak/3D-scanning-using-laser-triangulation/edit/main/image.png]

Description
This project implements a 3D scanning system using a camera and a line laser (laser triangulation) to reconstruct the shape of a pipe (or similar cylindrical object). It processes images of the laser line projected onto the object, calculates 3D points, fits the shape, and visualizes the results along with an ideal cylinder model.

This was developed as a personal project to explore 3D reconstruction techniques using computer vision and laser triangulation methods.

Features
Camera calibration loading (matrix and distortion coefficients).

Laser line extraction from images using image processing techniques (ROI, HSV filtering, thresholding, skeletonization).

Calculation of 3D points based on camera parameters and laser plane intersection.

Transformation of points from camera coordinates to world coordinates.

Robust arc/circle fitting on 3D point cloud slices with regularization.

Calculation of pipe radius and center position statistics.

3D visualization of the reconstructed point cloud and fitted cylinders (ideal, min/max measured) using Plotly.

Setup and Installation
Clone the repository:

bash
git clone https://github.com/Arashpirak/3D-scanning-using-laser-triangulation.git
cd 3D-scanning-using-laser-triangulation
Create a virtual environment (Recommended):

bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
pip install -r requirements.txt
Configuration
Calibration Files: Place your camera matrix (cameraMatrix.npy), distortion coefficients (distCoeffs.npy), and optimized transformation parameters (Matrixparameters.json) in the config/ directory (or update paths in the script). Ensure these files match the camera and setup used for capturing images.

Script Parameters: Key parameters (like Stepvalue, ideal_radius, camera specs, file paths, thresholds) are defined near the top of the laser_scanner_3d.py script. Modify them as needed for your setup.

folder_path: Important: Set this to the directory containing your input images.

ideal_radius: Set the expected radius of the pipe in mm.

Etc. (list other important parameters)

Input Images: Place the images (.jpg, .png) containing the laser line projected onto the object in the folder specified by folder_path. Sample images are provided in the data/ folder.

Usage
Run the main script from the project's root directory:

bash
python laser_scanner_3d.py
