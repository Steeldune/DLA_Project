# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import random
import numpy as np
import matplotlib.pyplot as plt


class ParticleBrown:
    def __init__(self, startCoords, endCoords, rand_len=512, r=1.0):
        self.dim = len(startCoords)
        self.pos = np.random.uniform(startCoords, endCoords)
        self.r = r
        for i in range(self.dim):
            self.pos[i] = random.uniform(startCoords[i], endCoords[i])
        self.d = np.zeros(self.dim)
        self.angle = np.zeros(self.dim - 1)
        self.rand_int = rand_len
        self.angle_array = np.array([self.dim - 1, self.rand_int])

    def gen_path(self):
        path = np.cumsum(self.d, axis=1)
        for i in range(self.dim):
            path[i] += self.pos[i]
        return path


class Brown2D(ParticleBrown):
    def __init__(self, startCoords, endCoords, rand_len=512, r=1):
        super().__init__(startCoords, endCoords, rand_len, r)
        self.dim = 2

    def gen_rand(self):
        self.angle_array = np.random.uniform(0, 2 * np.pi, self.rand_int)
        self.d = np.array([np.cos(self.angle_array) * self.r, np.sin(self.angle_array) * self.r])


class Brown3D(ParticleBrown):
    def __init__(self, startCoords, endCoords, rand_len=512, r=1):
        super().__init__(startCoords, endCoords, rand_len, r)
        self.dim = 3

    def gen_rand(self):
        self.angle_array = np.random.uniform([0, 0], [2 * np.pi, np.pi], [2, self.rand_int])
        self.d = np.array([self.r * np.sin(self.angle[1]) * np.cos(self.angle[0]),
                           self.r * np.sin(self.angle[1]) * np.sin(self.angle[0]), self.r * np.cos(self.angle[1])])


class DLATRee:
    def __init__(self, ini_coords):
        self.start_coods = ini_coords


def increment_particles(coords):
    random.random()
    return None


def check_tree(coords):
    return None


if __name__ == '__main__':
    origin = np.array([0.0, 0.0])
    final_coords = np.array([10.0, 10.0])
    George = DLATRee([5.0, 5.0])

    Jerry = Brown2D(origin, final_coords, rand_len=512, r=.3)
    print(Jerry.pos)
    Jerry.gen_rand()
    print(len(Jerry.angle_array))
    print(George.start_coods)
    path = Jerry.gen_path()

    plt.scatter(path[0], path[1], s=40, c='Red', alpha=0.5)
    plt.scatter(George.start_coods[0], George.start_coods[1], s=40, c='Blue', alpha=0.75)
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    plt.show()
