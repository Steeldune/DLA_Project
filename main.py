# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import random
import numpy as np
import matplotlib.pyplot as plt


class ParticleBrown:
    def __init__(self, startCoords, endCoords, rand_len=512):
        self.dim = len(startCoords)
        self.pos = np.random.uniform(startCoords, endCoords)
        self.r = 1
        for i in range(self.dim):
            self.pos[i] = random.uniform(startCoords[i], endCoords[i])
        self.d = np.zeros(self.dim)
        self.angle = np.zeros(self.dim - 1)
        self.rand_int = rand_len
        self.angle_array = np.array([self.dim - 1, self.rand_int])


class Brown2D(ParticleBrown):
    def __init__(self, startCoords, endCoords):
        super().__init__(startCoords, endCoords)
        self.dim = 2

    def gen_rand(self):
        self.angle_array = np.random.uniform(0, 2 * np.pi, self.rand_int)


    def calc_v(self):
        self.angle[0] = random.uniform(0, 2 * np.pi)
        self.d[0] = np.cos(self.angle) * self.r
        self.d[1] = np.sin(self.angle) * self.r


class Brown3D(ParticleBrown):
    def __init__(self, startCoords, endCoords):
        super().__init__(startCoords, endCoords)
        self.dim = 3

    def gen_rand(self):
        self.angle_array = np.random.uniform([0, 0], [2 * np.pi, np.pi], [2, self.rand_int])

    def calc_v(self):
        self.angle[0] = random.uniform(0, 2 * np.pi)
        self.angle[1] = random.uniform(0, np.pi)
        self.d[0] = self.r * np.sin(self.angle[1]) * np.cos(self.angle[0])
        self.d[1] = self.r * np.sin(self.angle[1]) * np.sin(self.angle[0])
        self.d[2] = self.r * np.cos(self.angle[1])


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

    Jerry = ParticleBrown(origin, final_coords)
    print(Jerry.pos)
    print(George.start_coods)

    plt.scatter(Jerry.pos[0], Jerry.pos[1], s=40, c='Red', alpha=0.5)
    plt.scatter(George.start_coods[0], George.start_coods[1], s=40, c='Blue', alpha=0.75)
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    plt.show()
