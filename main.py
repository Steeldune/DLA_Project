# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import random
import numpy as np
import matplotlib.pyplot as plt


class ParticleBrown:
    def init(self, startCoords, endCoords):
        self.dim = len(startCoords)
        self.poss = np.zeros(self.dim)
        self.r = 1
        for i in range(self.dim):
            self.poss[i] = random.uniform(startCoords[i], endCoords[i])
        return self.poss

    def increment(self):
        if self.dim == 2:
            self.angle = random.uniform(0, 2*np.pi)
            self.dx = np.cos(self.angle) * self.r
            self.dy = np.sin(self.angle) * self.r
        elif self.dim == 3:
            self.theta = random.uniform(0, 2*np.pi)
            self.phi = random.uniform(0, np.pi)
            self.dx = self.r * np.sin(self.phi) * np.cos(self.theta)
            self.dy = self.r * np.sin(self.phi) * np.sin(self.theta)
            self.dz = self.r * np.cos(self.phi)



class DLATRee:
    def init(self, ini_coords):
        self.start_coods = ini_coords




def increment_particles(coords):
    random.random()


def check_tree(coords):





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    origin = np.array([0.0,0.0])
    final_coords = np.array([10.0, 10.0])

    Jerry = ParticleBrown()
    print(Jerry.dim)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
