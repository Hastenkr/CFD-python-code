import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from PIL import Image
import io

# --------------- Mesh grid -----------------------
nel=20
ntstep=200
x=np.linspace(0, 1, nel+1)
y=np.zeros(nel+1)
t=np.linspace(0,15,ntstep+1)

# ------------------ Define fig to image ------------------
def fig2img(fig):
    buf=io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

# ----------------- initiate variables ---------------
A=np.zeros([nel,nel])

# -------------- Finite differences model ---------------

# ICs and BCs | Model variables
T0=40; BCs=np.array([0,-1])
T_old=np.ones(nel)*T0 # initial conditions
T_old[0]=100; T_old[-1]=20 # boundary conditions

h=1/nel # nodal distance
a=0.2  # thermal conductivity
e=0.5 # desirable temp convergence

delta=(h**2)/(2*a) # Courant-Friedrichs-Lewy condition

for i in range(nel):
    for j in range(nel):
        if i==j:
            A[(i,j)]=2
        elif abs(i-j)==1:
            A[(i,j)]=-1
        else:
            A[(i,j)]=0
A=A*(a/h**2) 

# explicit solution
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

list_gif=[]

T_new = (np.eye(nel) - A*delta) @ T_old # one time step forward
T_new[0] = 100; T_new[-1] = 20  
diffs=np.delete(abs(T_new-T_old),BCs).max()

while diffs>e:
    T_new = (np.eye(nel) - A*delta) @ T_old  # one time step forward
    T_new[0] = 100; T_new[-1] = 20  
    
    color_values = T_new
    fig, ax = plt.subplots()
    cmap = mpl.cm.jet

    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(color_values.min(), color_values.max()))
    lc.set_array(color_values) # Set the colors for each segment
    lc.set_linewidth(3)

    line = ax.add_collection(lc)

    fig.colorbar(line, ax=ax, label=r'Temperature', orientation='horizontal')

    ax.set_xlim(x.min()-0.2, x.max()+0.2)
    ax.set_ylim(-1, 1)
    
    plt.close(fig)
    img = fig2img(fig)
    list_gif.append(img)

    diffs=np.delete(abs(T_new-T_old),BCs).max()
    T_old=T_new

    
# -------------------- Plot and make gif --------------------
list_gif[0].save('temperature movie.gif',
                 save_all=True,append_images=list_gif[1:],optimize=False, duration=50, loop=0)
