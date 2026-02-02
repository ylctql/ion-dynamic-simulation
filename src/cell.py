import numpy as np  # version 1.17.2
from scipy.spatial import Delaunay  # version 1.4.1
import matplotlib.pyplot as plt  # version 3.1.2
import random


def create_data():
    '''n = 50
    x = [random.randint(1, 50) for _ in range(n)]
    y = [random.randint(51, 100) for _ in range(n)]
    points = [(i, j) for i, j in zip(x, y)]
    points = np.array(points)'''
    points = np.array([[1,0],[0, 0],[0.5,0.86]])
    return points


def create_delauney(points):
    # create a Delauney object using (x, y)
    tri = Delaunay(points)

    # paint a triangle
    plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), c='black')
    plt.plot(points[:, 0], points[:, 1], 'o', c='green')
    plt.axis('equal')
    plt.show()


def main():
    # create random data
    points = create_data()
    # create your Delauney using random data
    calc(points)
    create_delauney(points)
    

def calc(points):
    tri = Delaunay(points)
    edges = set([e for n in tri.simplices for e in [(n[0], n[1]), (n[1], n[2]), (n[2], n[0])]])
    l_edges = [((points[e[0]][0] - points[e[1]][0]) ** 2 + (points[e[0]][1] - points[e[1]][1]) ** 2) ** 0.5 for e in edges]
    print(np.mean(l_edges), np.std(l_edges))

def test():
    points = np.array([[1,1,0],[0,1,0],[0.5,1,0.86]])
    with open('test_points.txt', 'w') as f:
        for i in range(10):
            print(np.array([points[:, 0], points[:, 2]]).T, file=f)

if __name__ == '__main__':
    test()