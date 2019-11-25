#!/usr/bin/python3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def rot_mat(roll, pitch, yaw):
    D = np.array([[   np.cos(yaw),    np.sin(yaw),              0],
                  [  -np.sin(yaw),    np.cos(yaw),              0],
                  [             0,              0,              1]])

    C = np.array([[ np.cos(pitch),              0, -np.sin(pitch)],
                  [             0,              1,              0],
                  [ np.sin(pitch),              0,  np.cos(pitch)]])

    B = np.array([[             1,              0,              0],
                  [             0,   np.cos(roll),   np.sin(roll)],
                  [             0,  -np.sin(roll),   np.cos(roll)]])

    A = np.matmul(np.matmul(B,C),D)
    return A

def deg2rad(deg):
    return np.pi/180 * deg

def rad2deg(rad):
    return 180/np.pi * rad

class Camera:
    def __init__(self,x,y,z,roll,pitch,yaw,wfov,hfov):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.wfov = wfov
        self.hfov = hfov

    def mk_fov(self):
        rot_center = rot_mat(self.roll, self.pitch, self.yaw)
        rot_wsub = rot_mat(self.roll, self.pitch, self.yaw - self.wfov/2)
        rot_wadd = rot_mat(self.roll, self.pitch, self.yaw + self.wfov/2)
        rot_hsub = rot_mat(self.roll, self.pitch - self.hfov/2, self.yaw)
        rot_hadd = rot_mat(self.roll, self.pitch + self.hfov/2, self.yaw)

        line_step = np.linspace([0,0,0],[10,0,0],100).T

        line_center = np.matmul(rot_center, line_step)
        line_wsub = np.matmul(rot_wsub, line_step)
        line_wadd = np.matmul(rot_wadd, line_step)
        line_hsub = np.matmul(rot_hsub, line_step)
        line_hadd = np.matmul(rot_hadd, line_step)

        loc = np.array([self.x, self.y, self.z])

        """
        line_center = (line_center.T + loc).T
        line_wsub = (line_wsub.T + loc).T
        line_wadd = (line_wadd.T + loc).T
        line_hsub = (line_hsub.T + loc).T
        line_hadd = (line_hadd.T + loc).T
        """
        """
        assert all(line_hsub[0,:] == line_hadd[0,:])
        assert all(line_hsub[1,:] == line_hadd[1,:])
        assert all(-line_hsub[2,:] == line_hadd[2,:])

        assert all(line_wsub[0,:] == line_wadd[0,:])
        assert all(-line_wsub[1,:] == line_wadd[1,:])
        assert all(line_wsub[2,:] == line_wadd[2,:])
        """

        return line_center, line_wsub, line_wadd, line_hsub, line_hadd
    
    def plot(self, ax):
        i = 0
        colors = ['g','b','b','r','r']
        for line in self.mk_fov():
            xx,yy,zz = line
            ax.plot(xx,yy,zz,c=colors[i])
            i+=1

class Webcam(Camera):
    def __init__(self,x,y,z,roll,pitch,yaw):
        super(Webcam,self).__init__(x,y,z,roll,pitch,yaw,deg2rad(60),deg2rad(60*9/16))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

xs = [0]
ys = [0]
zs = [0]

line_step = np.linspace(0,10,100)
line_x = line_step
line_y = np.array([0]*100)
line_z = np.array([0]*100)
line_xyz = np.array([line_x,line_y,line_z])

print(rot_mat(0,np.pi/2,0).shape)
print(line_xyz.shape)

line_rot = np.matmul(rot_mat(0,np.pi/2,0),line_xyz)

ax.scatter(xs,ys,zs, c='r', marker='o')

#cam1 = Webcam(-2,-2,0,0,deg2rad(30),deg2rad(-45))
cam1 = Webcam(0,0,0,deg2rad(90),deg2rad(90),deg2rad(45))
#cam2 = Webcam(2,-2,0,0,deg2rad(30),deg2rad(-135))
#cam3 = Webcam(2,2,0,0,deg2rad(30),deg2rad(135))
cam1.plot(ax)
#cam2.plot(ax)
#cam3.plot(ax)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([-2,2])

plt.show()
