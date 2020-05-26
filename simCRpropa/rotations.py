import numpy as np


def car2sph(vec, axis = 0):
    """transform a cartesian vector into spherical coordinates"""
    result = np.zeros(vec.shape)
    result[0,:] = np.linalg.norm(vec, axis = axis) # r component
    #result[1,:] = np.arctan(vec[1,:] / vec[0,:]) # phi component
    result[1,:] = np.arctan2(vec[1,:] , vec[0,:]) # phi component
    result[2,:] = np.arccos(vec[2,:] / result[0,:]) # theta component
    return result


def project2observer(vec, unitvec2obs, axis = 0):
    """transforms vector to observer's coordinate system

    Parameter
    ---------
    vec: `~numpy.ndarray`
        Vector to be transformed, usually, this should be the momemtum vector
        at the time of detection or momentum vector at origin.

    unitvec2obs: `~numpy.ndarray`
        Unit vector between point where event hit the sphere (observer's position)
        and the point where event originated (the source).
        In CRPropa notation, this would be (X - X0) / || X - X0 ||

    {options}

    axis: int
        axis of vector components
        default: 0

    Returns
    -------
    `~numpy.ndarray` with vec projected onto new coordinate system
    of observer, where z-axis points along (X - X0)
    """
    r = np.linalg.norm(unitvec2obs, axis = axis)

    if axis == 0:
        rho = np.sqrt(unitvec2obs[0,:]*unitvec2obs[0,:] + \
                      unitvec2obs[1,:]*unitvec2obs[1,:])
        cosphi = unitvec2obs[0,:] / rho
        sinphi = unitvec2obs[1,:] / rho
        costheta = unitvec2obs[2,:] / r

    elif axis == 1:
        rho = np.sqrt(unitvec2obs[:,0]*unitvec2obs[:,0] + \
                      unitvec2obs[:,1]*unitvec2obs[:,1])
        cosphi = unitvec2obs[:,0] / rho
        sinphi = unitvec2obs[:,1] / rho
        costheta = unitvec2obs[:,2] / r

    sintheta = rho / r

    e1 = np.vstack([-sinphi, cosphi, np.zeros_like(sinphi)])
    e2 = np.vstack([cosphi * costheta, sinphi * costheta, -sintheta])
    e3 = -unitvec2obs

    result = np.vstack([np.sum(vec * e1, axis = 0),
                        np.sum(vec * e2, axis = 0),
                        np.sum(vec * e3, axis = 0)])
    return result

def projectjetaxis(vec, jet_opening_angle = 5.,
                    jet_theta_angle = 5.,
                    jet_phi_angle = 90.):
    """
    Project initial momentum vectors on jet axis
    and select only momentum vectors that fall within cone

    Parameters
    ----------
    vec: `~numpy.ndarray`
        Vector of initial momenta
    jet_opening_angle: float
        full jet opening angle (aperture) in degrees
    jet_theta_angle: float
        theta angle of jet axis, in degrees, 
        this is the angle to the l.o.s. to the observer
    jet_phi_angle: float
        phi angle of jet axis, in degrees

    Returns
    -------
    array with mask for initial momentum vectors
    """
    phi = np.radians(jet_phi_angle)
    theta = np.radians(jet_theta_angle)

    # jet vector in observers frame
    vecjet = np.vstack([np.ones(vec.shape[1]) * np.cos(phi) *np.sin(theta),
                        np.ones(vec.shape[1]) * np.sin(phi) *np.sin(theta),
                        np.ones(vec.shape[1]) * np.cos(theta)])

    # angle between jet axis and initial momentum
    cosangle = np.sum(vecjet * -vec, axis = 0)

    # restrict to those photons inside cone
    # cos(alpha) >= cos(theta_jet / 2.) is equal to alpha <= theta_jet / 2.
    return cosangle >= np.cos(np.radians(jet_opening_angle/2.))

# DEPRECATED FUNCTIONS:


def setRz(phi):
    """Rotation matrix around z axis"""
    Rz = np.zeros((phi.size,3,3))
    Rz[:,2,2] = 1.
    Rz[:,0,0] = np.cos(phi)
    Rz[:,1,1] = Rz[:,0,0]
    Rz[:,0,1] = -np.sin(phi)
    Rz[:,1,0] = -Rz[:,0,1]  
    return Rz

def setRy(phi):
    """Rotation matrix around y axis"""
    Ry = np.zeros((phi.size,3,3))
    Ry[:,1,1] = 1.
    Ry[:,0,0] = np.cos(phi)
    Ry[:,2,2] = Ry[:,0,0]
    Ry[:,0,2] = -np.sin(phi)
    Ry[:,2,0] = -Ry[:,0,2]
    return Ry

def setRx(phi):
    """Rotation matrix around x axis"""
    Rx = np.zeros((phi.size,3,3))
    Rx[:,0,0] = 1.
    Rx[:,1,1] = np.cos(phi)
    Rx[:,2,2] = Rx[:,0,0]
    Rx[:,1,2] = -np.sin(phi)
    Rx[:,2,1] = -Rx[:,0,2]
    return Rx

def setRyRz(phi_y,phi_z):
    """Rotate first around z and then around y axis"""
    if not phi_y.size == phi_z.size: 
        raise RuntimeError
    R = np.zeros((phi_y.size,3,3))
    R[:,0,0] = np.cos(phi_y) * np.cos(phi_z)
    R[:,0,1] = -np.cos(phi_y) * np.sin(phi_z)
    R[:,0,2] = -np.sin(phi_y)
    R[:,1,0] = np.sin(phi_z)
    R[:,1,1] = np.cos(phi_z)
    R[:,1,2] = np.zeros(phi_z.size)
    R[:,2,0] = np.cos(phi_z) * np.sin(phi_y)
    R[:,2,1] = -np.sin(phi_z) * np.sin(phi_y)
    R[:,2,2] = np.cos(phi_y)
    return R

def calcAlpha(rCar, rSph, D):
    """
    Calculate the angle for rotating the line of sights
    of the simulated photons to the observer

    Parameters
    ----------
    rCar: `~numpy.ndarray`
        (3,n) dim vector of the photon positions on the sphere as 
        in cartesian coordinates, used for the rotation

    rSph: `~numpy.ndarray`
        (3,n) dim vector of the photon positions on the sphere as 
        in spherical coordinates, used for the rotation

    D: float
        Source distance in Mpc. Assuming that the source is 
        on x-axis

    Returns
    -------
    tuple with rotation angles around y and z axis
    """
    rxyabs = np.linalg.norm(rCar[:-1,:], axis = 0)
    denominator_xy = np.sqrt(D**2. + rxyabs**2. \
                              -2. * D * rxyabs * np.cos(rSph[1,:]))
    denominator_xz = np.sqrt(D**2. + rSph[0,:]**2. \
                              -2. * D * rSph[0,:] * np.cos(np.pi/2. - rSph[-1,:]))
    #denominator_xz = np.sqrt(conf['Source']['D']**2. + rRotSph[:,0]**2. \
#                              -2. * conf['Source']['D'] * rRotSph[:,0] * np.cos(np.pi/2. - rRotSph[:,-1]))
    alpha_xy = np.arcsin(rxyabs * np.sin(rSph[1,:]) / denominator_xy)
    alpha_xz = np.arcsin(rSph[0,:] * np.sin(np.pi / 2. - rSph[-1,:]) / denominator_xz)
    #alpha_xz = np.arcsin(rRotSph[:,0] * np.sin(np.pi / 2. - rRotSph[:,-1]) / denominator_xz)
    return alpha_xy, alpha_xz

def calcRot(xCar, rCar, D):
    """
    Parameters
    ----------
    xCar: `~numpy.ndarray`
        (3,n) dim vector of the photon positions/momenta on the sphere as 
        in cartesian coordinates

    rCar: `~numpy.ndarray`
        (3,n) dim vector of the photon positions on the sphere as 
        in cartesian coordinates, used for the rotation

    D: float
        Source distance in Mpc. Assuming that the source is 
        on x-axis

    Returns
    -------
    tuple with rotated r,x vectors in Cartesian and Spherical ccordinates
    """
        
    rSph = car2sph(rCar, axis = 0)
    xSph = car2sph(xCar, axis = 0)

    alpha_xy, alpha_xz = calcAlpha(rCar, rSph, D)

    rRotCar = np.zeros_like(rCar)
    xRotCar = np.zeros_like(xCar)

    #for i in range(rRotCar.shape[1]):
    #    rRotCar[:,i] = np.dot(setRyRz(alpha_xz[i],alpha_xy[i]),rCar[:,i])
    #    xRotCar[:,i] = np.dot(setRyRz(alpha_xz[i],alpha_xy[i]),xCar[:,i])

    rRotCar = np.matmul(setRyRz(alpha_xz,alpha_xy),rCar.T[...,np.newaxis])[...,0].T
    xRotCar = np.matmul(setRyRz(alpha_xz,alpha_xy),xCar.T[...,np.newaxis])[...,0].T

    rRotSph = car2sph(rRotCar, axis = 0)
    xRotSph = car2sph(xRotCar, axis = 0)
    return rRotCar, rRotSph, xRotCar, xRotSph
