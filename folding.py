import numpy as np
import random 
import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D

# Some Variables to control number of amino acids and the dimensions of the grid, temperature, Boltzmann constant
num_acids = 50
dim = num_acids *3
T=5
kb = 1.38e-23
animate = True

# Storage variables
A = np.zeros((num_acids))
Axyz = np.zeros((num_acids, 3))
totalE = list()

# TODO: start folding and then tack on more acids as its in the middle of folding

# Generate the primary structure and give each acid coords
for i in range(len(A)):
    A[i]=random.randint(0,19)
    Axyz[i]=[i, 0, 0]
    
#Matrix of interaction energies
#Taken from: doi:10.1016/j.jmb.2005.01.071
J = [[-0.2,  -0.44,  0.16,  0.26, -0.46, -0.26,  0.5,  -0.57,  0.1,  -0.36, -0.22,  0.07,  0.14,  0.01,  0.20, -0.09, -0.05, -0.42,  0.05, -0.50],    
     [-0.44, -2.99,  0.21,  0.19, -0.88, -0.34, -1.11, -0.36, -0.09, -0.53, -0.43, -0.52, -0.14, -0.43, -0.24,  0.13, -0.22, -0.62,  0.24, -0.79],
     [ 0.16,  0.21,  0.17,  0.55,  0.38,  0.35, -0.23,  0.44, -0.39,  0.28,  0.35, -0.02,  1.03,  0.49, -0.37,  0.19, -0.12,  0.69,  0.04,  0.43],
     [ 0.26,  0.19,  0.55,  0.6,   0.55,  0.65,  0.18,  0.37, -0.47,  0.33,  0.29,  0.01,  0.69,  0.04, -0.52,  0.18,  0.37,  0.39,  0.03,  0.17],
     [-0.46, -0.88,  0.38,  0.55, -0.94,  0.17, -0.4,  -0.88,  0.01, -1.08, -0.78,  0.22,  0.20,  0.26, -0.19, -0.22,  0.02, -1.15, -0.60, -0.88],
     [-0.26, -0.34,  0.35,  0.65,  0.17, -0.12,  0.18,  0.24,  0.19,  0.24,  0.02, -0.04,  0.60,  0.46,  0.50,  0.28,  0.28,  0.27,  0.51, -0.35],
     [ 0.5,  -1.11, -0.23,  0.18, -0.4,   0.18,  0.42,  0,     0.79, -0.24, -0.07,  0.20,  0.25,  0.69,  0.24,  0.21,  0.11,  0.16, -0.85, -0.26],
     [-0.57, -0.36,  0.44,  0.37, -0.88,  0.24,  0,    -1.16,  0.15, -1.25, -0.58, -0.09,  0.36, -0.08,  0.14,  0.32, -0.27, -1.06, -0.68, -0.85],
     [ 0.1,  -0.09, -0.39, -0.47,  0.01,  0.19,  0.79,  0.15,  0.42,  0.13,  0.48,  0.26,  0.50,  0.15,  0.53,  0.10, -0.19,  0.10,  0.10,  0.04],
     [-0.36, -0.53,  0.28,  0.33, -1.08,  0.24, -0.24, -1.25,  0.13, -1.10, -0.50,  0.21,  0.42, -0.01, -0.07,  0.17,  0.07, -0.97, -0.95, -0.63],
     [-0.22, -0.43,  0.35,  0.29, -0.78,  0.02, -0.07, -0.58,  0.48, -0.50, -0.74,  0.32,  0.01,  0.26,  0.15,  0.48,  0.16, -0.73, -0.56, -1.02],
     [ 0.07, -0.52, -0.02,  0.01,  0.22, -0.04,  0.2,  -0.09,  0.26,  0.21,  0.32,  0.14,  0.27,  0.37,  0.13,  0.15,  0.10,  0.40, -0.12,  0.32],
     [ 0.14, -0.14,  1.03,  0.69,  0.2,   0.60,  0.25,  0.36,  0.5,   0.42,  0.01,  0.27,  0.27,  1.02,  0.47,  0.54,  0.88, -0.02, -0.37, -0.12],
     [ 0.01, -0.43,  0.49,  0.04,  0.26,  0.46,  0.69, -0.08,  0.15, -0.01,  0.26,  0.37,  1.02, -0.12,  0.24,  0.29,  0.04, -0.11,  0.18,  0.11],
     [ 0.2,  -0.24, -0.37, -0.52, -0.19,  0.5,   0.24,  0.14,  0.53, -0.07,  0.15,  0.13,  0.47,  0.24,  0.17,  0.27,  0.45,  0.01, -0.73,  0.01],
     [-0.09,  0.13,  0.19,  0.18, -0.22,  0.28,  0.21,  0.32,  0.10,  0.17,  0.48,  0.15,  0.54,  0.29,  0.27, -0.06,  0.08,  0.12, -0.22, -0.14],
     [-0.05, -0.22, -0.12,  0.37,  0.02,  0.28,  0.11, -0.27, -0.19,  0.07,  0.16,  0.10,  0.88,  0.04,  0.45,  0.08, -0.03, -0.01,  0.11, -0.32],
     [-0.42, -0.62,  0.69,  0.39, -1.15,  0.27,  0.16, -1.06,  0.10, -0.97, -0.73,  0.40, -0.02, -0.11,  0.01,  0.12, -0.01, -0.89, -0.56, -0.71],
     [ 0.05,  0.24,  0.04,  0.03, -0.6,   0.51, -0.85, -0.68,  0.10, -0.95, -0.56, -0.12, -0.37,  0.18, -0.73, -0.22,  0.11, -0.56, -0.05, -1.41],
     [-0.5,  -0.79,  0.43,  0.17, -0.88, -0.35, -0.26, -0.85,  0.04, -0.63, -1.02,  0.32, -0.12,  0.11,  0.01, -0.14, -0.32, -0.71, -1.41, -0.76]]

#Are A1 and A2 nearest neighours?
def NN(A1, A2):
    areNN=False
    dA = np.zeros(len(A1)) #difference in xyz coords
    
    for i in range(0,len(A1)):
        dA[i] = np.abs(A1[i] - A2[i])
    
    #np.all ensures the arrays are exactly equal
    #acids are nearest neighbours if their distance is exactly 1, since we are using a cubic lattice, this means that the x,y, or z must be exactly one off
    if(np.all(dA == [1,0,0]) or np.all(dA == [0,1,0]) or np.all(dA == [0,0,1])):
        areNN=True
    
    return areNN
    
#Find the energy of the current chain
def calcE(A,Axyz):
    E = 0
    for i in range(len(A)):
        for j in range(len(A)):
            #are i and j nearest neighbours and not bonded?
            if(not (np.abs(i-j) in [0,1]) and NN(Axyz[i], Axyz[j])):
                E += J[int(A[i])][int(A[j])]
    return E

#Some variables to keep track of events
n=0
rnnfail=0
lnnfail=0
efail=0
nallowed=0
totalE.append(calcE(A,Axyz))
lastmove=0
while(nallowed<500):
    n+=1 #step counter
    if(n-lastmove > 5000):
        break
    
    #calculate E1
    E1 = calcE(A,Axyz)
    
    #generate move
    moves = [[1,1,0],[0,1,1],[1,0,1],[-1,1,0],[0,-1,1],[-1,0,1],[1,-1,0],[0,1,-1],[1,0,-1],[-1,-1,0],[0,-1,-1],[-1,0,-1]]
    
    Ai = random.randint(0, len(A)-1) #pick random acid
    Af = Axyz[Ai] + moves[random.randint(0,11)] #pick a random move
    
    #test if move allowed
    allowed = True
    
    #Af must be nearest neighbours with both its preceeding and succeeding acid to not break the peptide bonds
    if(not Ai==0):
        if(not NN(Af,Axyz[Ai-1])):
            allowed = False
            lnnfail+=1
    if(not Ai==len(A)-1):
        if(not NN(Af,Axyz[Ai+1])):
            allowed = False
            rnnfail+=1
    
    #Af must not be an already occupied position
    for i in range(len(A)):
        if(np.all(Af == Axyz[i])):
            allowed = False
            efail+=1
    
    #calculate E2 and deltaE
    if(allowed):
        newAxyz = Axyz.copy()
        newAxyz[Ai] = Af
        E2 = calcE(A, newAxyz)
        deltaE = E2-E1
    else:
        deltaE = 0
        newAxyz = Axyz.copy()

    
    #move if deltaE is negative or less than a random boltzmann energy
    bE = np.exp((-1 * deltaE)/(kb * T))
    if(allowed and (deltaE <= 0 or random.uniform(0,1) < bE)):
        lastmove=n
        Axyz = newAxyz.copy()
        nallowed+=1
        totalE.append(E2)
        
        if(animate):
            fig = plt.figure(figsize=(4,4), dpi=600)
            ax = fig.add_subplot(111, projection='3d')
            ax.set(xlim=(-1, len(A)), ylim=(-5, 5), zlim=(-5,5))


            for i in range(1,len(A)):
                dx = np.linspace(newAxyz[i-1][0], newAxyz[i][0], 1000)
                dy = np.linspace(newAxyz[i-1][1], newAxyz[i][1], 1000)
                dz = np.linspace(newAxyz[i-1][2], newAxyz[i][2], 1000)
                ax.scatter(dx,dy,dz,c='k')

            ax.scatter([b[0] for b in newAxyz],[b[1] for b in newAxyz],[b[2] for b in newAxyz],c='r')
            name="pics/fold-{:04d}.png".format(nallowed)
            fig.savefig(name, bbox_inches='tight')
            plt.close(fig)    # close the figure so its not kept in ram
    
# Summary
print("Number of overlapping states denied: " + str(efail))
print("Number of bond stretches denied: {} (right) {} (left)".format(rnnfail,lnnfail))
print("Number of folds made: " + str(nallowed))
print("Number of steps: " + str(n))

# Final plot 
fig = plt.figure(figsize=(4,4), dpi=600)
ax = fig.add_subplot(111, projection='3d')

for i in range(1,len(A)):
    dx = np.linspace(newAxyz[i-1][0], newAxyz[i][0], 1000)
    dy = np.linspace(newAxyz[i-1][1], newAxyz[i][1], 1000)
    dz = np.linspace(newAxyz[i-1][2], newAxyz[i][2], 1000)
    ax.scatter(dx,dy,dz,c='k')

ax.scatter([b[0] for b in newAxyz],[b[1] for b in newAxyz],[b[2] for b in newAxyz],c='r')
ax.set(xlim=(-1, 10), ylim=(-5, 5), zlim=(-5,5))

plt.show()

# Energy plot
fig2 = plt.figure(figsize=(4,4), dpi=600)
ax2 = plt.axes()
ax2.plot(range(0, nallowed+1,1), totalE)
plt.show()
fig2.savefig("EPlot.png", bbox_inches='tight')