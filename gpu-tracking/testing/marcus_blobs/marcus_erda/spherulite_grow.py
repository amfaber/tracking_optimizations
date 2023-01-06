import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import seaborn as sns

def distance_to_point(x, y, z, x0, y0, z0):
    return np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)

distance_to_point_vec = np.vectorize(distance_to_point)

def return_gaussian(points, point,threshold=0.1):
    #pick the last 50 points, hopefully they are representative of the distribution
    points = points[-50:]
    x, y, z = points.T
    x0, y0, z0 = point
    distances = distance_to_point_vec(x, y, z, x0, y0, z0)
    if threshold is not None:
        distancesfmt = distances[distances < threshold]
    else:
        distancesfmt = distances
    gaussian = np.sum(np.exp(-distancesfmt**2))
    return gaussian

def calculate_probability(points, threshold=0.1):
    points = np.array(points)
    probas = []
    for point in points:
        deltaxyz = np.random.normal(0, 0.1, 3)
        point_probe = point + deltaxyz
        point_probability = return_gaussian(points, point_probe, threshold=threshold)
        probas.append(point_probability)
    
    argmin = np.argmin(probas)
    points = np.append(points, [points[argmin] + deltaxyz], axis=0)
    return points

def main(threshold=0.1,n_points=100):
    points = np.array([[0, 0, 0]])
    for i in range(n_points):
        points = calculate_probability(points, threshold=threshold)
    return points

def animate3d(points):
    fig = plt.figure()
    for i in range(100):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:i, 0], points[:i, 1], points[:i, 2])
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        plt.pause(0.0001)
        plt.clf()

def animate2d(points):
    fig, ax = plt.subplots()
    for i in range(100):
        ax.scatter(points[:i, 0], points[:i, 1])
        ax.set(xlim=(-1, 1), ylim=(-1, 1))
        plt.pause(0.0001)
        plt.cla()

def compare_growth(n_points,num_plots):
    fig, ax = plt.subplots(2,num_plots, figsize=(20,8))
    for i in range(num_plots):
        thresh = 0.15*(i+1)
        #thresh = 0.35
        points = main(threshold=thresh,n_points=n_points)
        color = np.linspace(0,n_points,n_points)
        ax[0,i].scatter(points[:n_points, 0], points[:n_points, 1], c = color, s=3)
        ax[0,i].set(xlim=(-3, 3), ylim=(-3, 3))
        ax[0,i].set_title('Threshold = {} '.format(np.round(thresh,2)))
        ax[1,i] = sns.kdeplot(points[:,0], points[:,1], ax=ax[1,i])
        ax[1,i].set(xlim=(-3, 3), ylim=(-3, 3))
        ax[1,i].set_title('Threshold = {} '.format(np.round(thresh,2)))
    runid = np.random.randint(1000)
    savestr = 'spherulite_growth_{},{},{}.png'.format(n_points,thresh,runid)
    plt.savefig(savestr, dpi=300, bbox_inches='tight')
    plt.show()

compare_growth(250,5)

# for thresh in [0.1,0.3,0.5,0.7,0.9]:
#     points = main(threshold=thresh*3,n_points=100)
#     animate2d(points)
#     plt.close()
