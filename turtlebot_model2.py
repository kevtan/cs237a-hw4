import numpy as np
import math

EPSILON_OMEGA = 1e-3

def compute_dynamics(x, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                        x: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to x.
        Gu: np.array[3,2] - Jacobian of g with respect ot u.
    """
    ########## Code starts here ##########
    cth = math.cos(x[2])
    sth = math.sin(x[2])
    cth_n = math.cos(x[2] + u[1] * dt)
    sth_n = math.sin(x[2] + u[1] * dt)
    if abs(u[1]) > EPSILON_OMEGA:
        r = u[0] / u[1]
        g = x + np.array([r*(sth_n - sth), -1*r*(cth_n - cth), u[1]*dt])
        Gx = np.array([
            [1, 0, r*(cth_n - cth)],
            [0, 1, r*(sth_n - sth)],
            [0, 0, 1]
            ])
        Gu = np.array([
            [(sth_n - sth)/u[1], -1*r*(sth_n - sth)/u[1] + r*dt*cth_n],
            [-1*(cth_n - cth)/u[1], r*(cth_n - cth)/u[1] + r*dt * sth_n],
            [0, dt]
        ])
    else:
        g = x + dt*np.array([u[0]*cth_n, u[0]*sth_n, 0])
        Gx = np.array([
            [1, 0, -1*u[0] * sth * dt],
            [0, 1, u[0] * cth * dt],
            [0, 0, 1]
        ])
        Gu = np.array([
            [cth*dt,  -1.0/2.0 * dt**2 * u[0] * sth_n],
            [sth*dt, 1.0/2.0 * dt**2 * u[0] * cth_n],
            [0, dt]
        ])

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line
    ########## Code starts here ##########
    # compute alpha_cam, alpha in camera coordinates
    alpha_cam = alpha - x[2] - tf_base_to_camera[2]

    # compute r_cam, r in camera coordinates
    # first compute cam_world, the x/y coordinates of the camera in world cartesian coordinates
    tf = np.array([
        [math.cos(x[2]), -1*math.sin(x[2])],
        [math.sin(x[2]), math.cos(x[2])]
    ])

    cam_world = x[0:2] + np.dot(tf, tf_base_to_camera[0:2])

    # next compute a, b, c in the equation ax + by = c, x cos alpha + y sin alpha = r which represents the line defined by alpha, r
    a = -1*math.cos(alpha)
    b = -1*math.sin(alpha)
    c = r

    # compute the min distance between cam_world and ax + by = c using (a*cam_world[0] + b*cam_world[1] + c) / sqrt(a^2 + b^2)
    # formula from https://www.geeksforgeeks.org/perpendicular-distance-between-a-point-and-a-line-in-2-d/
    r_cam = (a*cam_world[0] + b*cam_world[1] + c) / math.sqrt(a**2 + b**2)

    h = np.array([alpha_cam, r_cam])

    # compute H
    abn = math.sqrt(a**2 + b**2)
    sth = math.sin(x[2])
    cth = math.cos(x[2])
    x_b = tf_base_to_camera[0]
    y_b = tf_base_to_camera[1]
    dx_cam_dth = -1 * sth * x_b - y_b * cth
    dy_cam_dth = x_b * cth - y_b * sth
    dr_dth = (a * dx_cam_dth + b * dy_cam_dth) / abn

    Hx = np.array([
        [0, 0, -1],
        [a / abn, b / abn, dr_dth]
    ])
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
