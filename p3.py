
import numpy as np
import matplotlib.pyplot as plt

x_new = np.linspace(0, 20, 21)
y_new = np.linspace(0, 20, 21)
z_new = np.linspace(0, 4, 5)

X, Y, Z = np.meshgrid(x_new, y_new, z_new)


def plot_grid(X, Y, Z, Xt=None, Yt=None, Zt=None, t=False):
 
    fig = plt.figure()
    fig = fig.add_subplot(111, projection='3d')
    if t:
        fig.scatter(X, Y, Z, c='k')
        fig.scatter(Xt, Yt, Zt, c='r')
    else:
        fig.scatter(X, Y, Z, c='r')

    plt.show()
    


def translate(X, Y, Z, v):
    return X + v[0], Y + v[1], Z + v[2]



def rotate(X, Y, Z, axis, angle):

    angle_rad = np.radians(angle)  
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)

    if axis == "x":
        Y1 = Y * cos - Z * sin
        Z1 = Y * sin + Z * cos
        return X, Y1, Z1
    elif axis == "y":
        X1 = X * cos + Z * sin
        Z1 = X * sin - Z * cos
        return X1, Y, Z1
    elif axis == "z":
        X1 = X * cos - Y * sin
        Y1 = X * sin + Y * cos
        return X1, Y1, Z

    


def rigid_transform(theta, omega, phi, p, q, r):
    v = np.array([p, q, r])
    X_new, Y_new, Z_new = rotate(X, Y, Z, "x", theta) 
    plot_grid(X,Y,Z,X_new, Y_new, Z_new,t=True)
    
    X_new, Y_new, Z_new = rotate(X, Y, Z, "y", omega) 
    plot_grid(X,Y,Z,X_new, Y_new, Z_new,t=True)
    
    X_new, Y_new, Z_new = rotate(X, Y, Z, "z", phi) 
    plot_grid(X,Y,Z,X_new, Y_new, Z_new,t=True)
    
    X_new, Y_new, Z_new = translate(X, Y, Z,v)
    plot_grid(X,Y,Z,X_new, Y_new, Z_new,t=True)





def affine_transform(s, theta, omega, phi, p, q, r):
    v = np.array([p, q, r])
    X_new, Y_new, Z_new = rotate(X, Y, Z, "x", theta) 
    plot_grid(X,Y,Z,X_new*s, Y_new*s, Z_new*s,v,t=True)
    
    X_new, Y_new, Z_new = rotate(X, Y, Z, "y", omega) 
    plot_grid(X,Y,Z,X_new*s, Y_new*s, Z_new*s,v,t=True)
    
    X_new, Y_new, Z_new = rotate(X, Y, Z, "z", phi) 
    plot_grid(X,Y,Z,X_new*s, Y_new*s, Z_new*s,v,t=True)
    
    X1, Y1, Z1 = translate(X, Y, Z,v)
    plot_grid(X,Y,Z,X_new*s, Y_new*s, Z_new*s,v,t=True)
  


plot_grid(X, Y, Z)
rigid_transform(30,30,10,4,3,2)
# affine_transform(0.5,30,30,10,4,3,2)
