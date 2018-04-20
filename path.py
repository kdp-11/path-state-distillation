import matplotlib as mpl
from qutip import *
import numpy as np
from math import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import pyplot
import itertools
import matplotlib.gridspec as gridspec

#prints full matrix instead of partial parts
np.set_printoptions(threshold=np.inf)

#np.set_printoptions(formatter={'all': '{:,}'.format})
#bloch sphere diagram, not bloch3d because of vtk version clash with mayavi
c=Bloch()

#|0> and |1> in 2D Hilbert space
a = basis (2,0)
b = basis(2,1)       

#logical zero after application of 16 generators pf [[5,1,3]] stabilizer code
zero = 1/4 * (tensor(a,a,a,a,a) + tensor(b,a,a,b,a) + tensor(a,b,a,a,b) + tensor(b,a,b,a,a) + tensor(a,b,a,b,a)  + tensor(a,a,b,a,b) - (tensor(b,b,b,b,a) + tensor(a,b,b,b,b) + tensor(b,a,b,b,b) + tensor(b,b,a,b,b) + tensor(b,b,b,a,b)  )- ( tensor(a,b,b,a,a) + tensor(a,a,b,b,a) + tensor(a,a,a,b,b) + tensor(b,a,a,a,b) + tensor(b,b,a,a,a) ))

#-------------square root of 1/16 to normalize-------------#
#logical one
one = 1/4 *(tensor(b,b,b,b,b) + tensor(a,b,b,a,b) + tensor(b,a,b,b,a) + tensor(a,b,a,b,b) + tensor(b,a,b,a,b)  + tensor(b,b,a,b,a) - (tensor(a,a,a,a,b) + tensor(b,a,a,a,a) + tensor(a,b,a,a,a) + tensor(a,a,b,a,a) + tensor(a,a,a,b,a)  )- ( tensor(b,a,a,b,b) + tensor(b,b,a,a,b) + tensor(b,b,b,a,a) + tensor(a,b,b,b,a) + tensor(a,a,b,b,b) ))

projector = (zero*zero.dag() +  one*one.dag()) #not normalized, trace is 32, thus divided by 32

#p3 = projector - projector.dag()
#print (p2)
#print (projector)
#print (p3) #---test for projector trace----

'''Distillation'''
def distill(list):
    #-------------ARRAY---------------------#
    x = list[0]
    y = list[1]
    z = list[2]
     #-----------ENCODE---------------------#
    rho  =  0.5*(qeye(2) + x*sigmax() + y*sigmay() + z*sigmaz() )
    rho_five = tensor(rho,rho,rho,rho,rho)
    encoder = projector*rho_five*projector #step that needs to be normalized
    encoder = encoder*(1/encoder.tr())

    #------------DECODE-------------------#
    alpha1 = np.array((zero.dag()*encoder*zero).full())
    beta1  = np.array((one.dag()*encoder*one).full())
    gamma1 = np.array((zero.dag()*encoder*one).full())
    delta1 = np.array((one.dag()*encoder*zero).full())

    alpha = alpha1[0][0]       #------COEFFICIENTS----------------------------------#
    beta  = beta1[0][0]        #------OF RHO'---------------------------------------#      
    gamma = gamma1[0][0]       #------INDEXING TWICE TO GET THE VALUES--------------#
    delta = delta1[0][0]

    rho_p = Qobj([[alpha, delta],[gamma, beta]]) #-------CONVERTING INTO 2X2 MATRIX WHICH IS RHO'----# 
                                                 #-------AND THEN INTO QUTIP OBJECT------------------#

    x_p = (rho_p*sigmax()).tr()      #----------SOLVING FOR X',Y',Z'---------------#
    y_p = (rho_p*sigmay()).tr()      #-----------------IN RHO'---------------------#
    z_p = (rho_p*sigmaz()).tr()

    return [x_p,y_p,z_p]            #---------RETURNING A DISTILLED ARRAY----------#
 
#-----------------FUNCTION TO MAKE THE PATH OF DISTILLATION FOR EACH POINT--------------------------#
def makepath(init,iters):#---------INIT IS INITIAL POINT WHILE ITERS IS NUMBER OF ITERATIONS------#
    point = init
    path = [point]
    for i in range (iters):
        point = distill(point)
        path  = np.append(path, [point], axis=0)
    return path

#mk1 = makepath([0.6,0.5,0.1],10)
#mk2 = makepath([0.8,0.4,0.1],10)
#mk3 = makepath([0.65,0,0.6],10)

#print (mk1)

''' Create grid of points'''
def pointgrid(n,a,b,c): #-------n = NUMBER OF POINTS ON X AND Y AXIS-----------#
    xx = np.linspace(0.4,a, num=n)
    yy = np.linspace(0.4,b,num=n)
    zz = np.linspace(0.3,c,num=n)
    xxyy = np.array([xx,yy,zz])
    pnt = [] 

    for i in itertools.product(*xxyy):
        pnt.append(i) #arrays within array created    
    point = np.array(pnt) #converted to numpy
           
    return point

#print (pointgrid(5,1,1,1))

'''Check if generated point within sphere'''
def ifwithinsphere(point):
    return sqrt( (point[0])**2 +(point[1])**2+ (point[2])**2 )

'''Run makepath of the points that are within the sphere. Create an array'''
paths= []
def path(num,a,b,c):
    grid = pointgrid(num,a,b,c)
    for i in range(len(grid)):
        if ifwithinsphere(grid[i]) < 1:
            path = makepath(grid[i],20)
            paths.append(path)
    return paths

paths = path(2,0.45,0.45,0.3)
print ((len(paths)))

#print (paths)

#x = paths[3][:,1]
#y = paths[3]#[:,1]
#z = paths[3]#[:,2]
#print (x)#,y,z)

xa = []
for i in range(len(paths)):
    x2 = paths[i][:,0]
    xa.append(x2)
x = np.array(xa)

ya = []
for i in range(len(paths)):
    y2 = paths[i][:,1]
    ya.append(y2)
y = np.array(ya)

za = []
for i in range(len(paths)):
    z2 = paths[i][:,2]
    za.append(z2)
z = np.array(za)

#T state
xT = sqrt(1/3)
yT = sqrt(1/3)
zT = sqrt(1/3)

#H state
xH = sqrt(1/2)
yH = 0
zH = sqrt(1/2)

#Wireframe of Sphere
fig = pyplot.figure()
ax = fig.gca(projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
xx = 1 * np.outer(np.cos(u), np.sin(v))
yy = 1 * np.outer(np.sin(u), np.sin(v))
zz = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
#ax.plot_wireframe(xx, yy, zz, color='0.45', linewidth=0.09)

#plotting of H and T states
ax.scatter(xT,yT,zT, "-o", color='#ff1a1a')
#ax.scatter(xH,yH,zH,"-o", color='#3366cc')

#plotting of paths
for i in range(len(paths)):
    ax.plot(x[i][::2],y[i][::2],z[i][::2], color = '#076443', linewidth = 0.4)


plt.title('Paths')
plt.ylabel('Y')
plt.xlabel('X')
#plt.legend()
pyplot.show()
