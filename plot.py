import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

class plot_3D:
    def __init__(self, V = (100, 100, 100), alpha = 0.2, style = 'default'):
        if style == 'xkcd':
            plt.xkcd()
        else:
            plt.style.use(style)
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        
        self.alpha = alpha
        self.V = V
        self.ax.set_xlim(0, V[1])
        self.ax.set_ylim(V[0] ,0)
        self.ax.set_zlim(0, V[2])
        self.ax.set_xlabel('Y axis')
        self.ax.set_ylabel('X axis')
        self.ax.set_zlabel('Z axis')
        
        scale_x = V[1]*2/(V[0]+V[1]+V[2])
        scale_y = V[0]*2/(V[0]+V[1]+V[2])
        scale_z = V[2]*2/(V[0]+V[1]+V[2])
        self.ax.get_proj = lambda: np.dot(Axes3D.get_proj(self.ax), np.diag([scale_x, scale_y, scale_z, 1]))
        
        self.boxes_num = 0
        self.boxes_df = pd.DataFrame(columns=['box','min_coord','max_coord','sides','color'])
    
    def add_box(self, v1, v2, mode = 'vector'):
        min_coord = np.array(list(v1))
        v2 = np.array(list(v2))
           
        if mode == 'vector':
            sides = v2
            max_coord = [v1[i] + sides[i] for i in range(3)]
        elif mode == 'EMS':
            sides = [v2[i] - v1[i]  for i in range(3)]
            max_coord = v2
        
        y_range, x_range, z_range = zip(min_coord, max_coord)
        
        # random color
        color = np.random.rand(3,)
        
        # Draw 6 Faces
        xx, xy, xz = np.meshgrid(x_range, y_range, z_range) # X
        yy, yx, yz = np.meshgrid(y_range, x_range, z_range) # Y
        zx, zz, zy = np.meshgrid(x_range, z_range, y_range) # Z
        for i in range(2):  
            self.ax.plot_wireframe(xx[i], xy[i], xz[i], color=color)
            self.ax.plot_surface(xx[i], xy[i], xz[i], color=color, alpha=self.alpha)
            self.ax.plot_wireframe(yx[i], yy[i], yz[i], color=color)
            self.ax.plot_surface(yx[i], yy[i], yz[i], color=color, alpha=self.alpha)
            self.ax.plot_wireframe(zx[i], zy[i], zz[i], color=color)
            self.ax.plot_surface(zx[i], zy[i], zz[i], color=color, alpha=self.alpha)
        
        # Record
        self.boxes_df = self.boxes_df.append({'box':self.boxes_num, 'sides': sides,
                                              'min_coord':min_coord, 'max_coord': max_coord,
                                              'color':color}, ignore_index=True)
        self.boxes_num += 1
    
    def findOverlapping(self, verbose = True):
        isOverlapped = False
        for A, B in combinations(range(self.boxes_num), 2):
            if self.intersection(A, B):
                isOverlapped = True
                if verbose:
                    print(f'Box {A} overlapped with box {B}.')
        if not isOverlapped:
            print('No overlapping boxes.')
        return
    
    def intersection(self, A, B):
        A1, A2 = self.boxes_df.loc[A,['min_coord', 'max_coord']].apply(lambda x : np.array(x))
        B1, B2 = self.boxes_df.loc[B,['min_coord', 'max_coord']].apply(lambda x : np.array(x))
        
        if np.all(A2 > B1) and np.all(A1 < B2):
            return True
        return False
    
    def show(self, elev = None, azim = None):
        self.ax.view_init(elev=elev, azim=azim)
        plt.show()

def plot_history(history, tick = 2):
    for target in ['mean', 'min']:
        plt.plot(history[target], label = target)
    plt.title('Fitness during evolution')
    plt.ylabel('fitness')
    plt.xlabel('generation')
    plt.xticks(np.arange(0, len(history['min']), tick))
    plt.legend()
    # h-line for integer
    for i in np.arange(math.ceil(min(history['min'])), int(max(history['mean']))+1):
        plt.axhline(y = i, color = 'g', linestyle = '-') 
    plt.show()
    
def draw_3D_plots(decoder, V):
    for i in range(decoder.num_opend_bins):
        container = plot_3D(V=V)
        for box in decoder.Bins[i].load_items:
            container.add_box(box[0], box[1], mode = 'EMS')
        print('Container',i, ':')
        container.findOverlapping()
        container.show()