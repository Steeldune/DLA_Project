# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm


class ParticleBrown:
    def __init__(self, startCoords, endCoords, rand_len=512, r=1.0):
        self.dim = len(startCoords)
        self.pos = np.zeros((self.dim, rand_len), dtype=float)
        self.r = r
        self.d = np.zeros(self.dim)
        self.angle = np.zeros(self.dim - 1)
        self.rand_int = rand_len
        self.angle_array = np.array([self.dim - 1, self.rand_int])
        self.path = np.zeros((self.dim, rand_len))

    def gen_path(self):
        self.path = np.cumsum(self.d, axis=1)
        for i in range(self.dim):
            self.path[i] += self.pos[i]
        return self.path

    def start_pos_rect(self, start_rect, end_rect):
        for i in range(self.dim):
            self.pos[i] = np.random.uniform(start_rect[i], end_rect[i])


class Brown2D(ParticleBrown):
    def __init__(self, startCoords, endCoords, rand_len=512, r=1, spawn_r=0):
        super().__init__(startCoords, endCoords, rand_len, r)
        self.dim = 2
        self.pos_theta = 0.0

    def gen_rand(self):
        self.angle_array = np.random.uniform(0, 2 * np.pi, self.rand_int)
        self.d = np.array([np.cos(self.angle_array) * self.r, np.sin(self.angle_array) * self.r])

    def start_pos_circle(self, spawn_r=0.0):
        self.pos_theta = np.random.uniform(0, 2 * np.pi)
        self.pos[0] = np.cos(self.pos_theta) * spawn_r
        self.pos[1] = np.sin(self.pos_theta) * spawn_r


class Brown3D(ParticleBrown):
    def __init__(self, startCoords, endCoords, rand_len=512, r=1):
        super().__init__(startCoords, endCoords, rand_len, r)
        self.dim = 3
        self.pos_theta = 0.0
        self.pos_phi = 0.0

    def start_pos_sphere(self, spawn_r=0.0):
        self.pos_theta = np.random.uniform(0, 2 * np.pi)
        self.pos_phi = np.arccos(np.random.uniform(-1, 1))

        self.pos[0] = spawn_r * np.sin(self.pos_phi) * np.cos(self.pos_theta)
        self.pos[1] = spawn_r * np.sin(self.pos_phi) * np.sin(self.pos_theta)
        self.pos[2] = spawn_r + np.cos(self.pos_phi)

    def gen_rand(self):
        self.angles_theta = np.random.uniform(0, 2 * np.pi, self.rand_int)
        self.angles_phi = np.arccos(np.random.uniform(-1, 1, self.rand_int))
        self.d = np.array([self.r * np.sin(self.angles_phi) * np.cos(self.angles_theta),
                           self.r * np.sin(self.angles_phi) * np.sin(self.angles_theta),
                           self.r * np.cos(self.angles_phi)])


class DLATree:
    def __init__(self, ini_coords, r=0.1):
        self.seed_array = np.array(ini_coords)
        self.r = r
        self.link_tree = []
        self.connect_node = 0
        self.link_fig = plt.figure()
        self.link_ax = self.link_fig.add_subplot()

    def check_path(self, path):
        check_indices = -1
        self.connect_node = -1
        for k, seed in enumerate(self.seed_array):
            temp_path = np.zeros((len(seed), len(path[0])))
            for i in range(len(seed)):
                temp_path[i] = path[i] - seed[i]
            temp_path = np.linalg.norm(temp_path, axis=0)
            check_path = np.where(abs(temp_path) < self.r, True, False)
            seed_indices = np.where(check_path == 1)[0]
            if len(seed_indices) > 0:
                if check_indices == -1:
                    check_indices = seed_indices[0]
                    self.connect_node = k
                elif seed_indices[0] < check_indices:
                    check_indices = seed_indices[0]
                    self.connect_node = k
        if self.connect_node > -1:
            self.link_tree.append([self.connect_node, np.shape(self.seed_array)[0]])
        return check_indices

    def add_seed(self, coords):
        self.seed_array = np.append(self.seed_array, [coords], axis=0)
        return len(self.seed_array)

    def gen_link_graph(self, colors):
        for i in range(len(self.link_tree)):
            self.link_ax.scatter(self.link_tree[i][1], self.link_tree[i][0], color=colors)

    def adjust_origin(self):
        mean_temp = np.mean(self.seed_array, axis=0)
        self.seed_array = np.subtract(self.seed_array, mean_temp)


class DLA3DTree(DLATree):
    def __init__(self, ini_coords, r=0.1):
        super().__init__(ini_coords, r)
        self.tree_fig = plt.figure()
        self.tree_ax = self.tree_fig.add_subplot(projection='3d')

    def gen_tree_graph(self, colors):
        for i in range(len(self.seed_array)):
            self.tree_ax.scatter(self.seed_array[i][0], self.seed_array[i][1], self.seed_array[i][2], s=30,
                                 color=next(colors), alpha=1)


if __name__ == '__main__':
    spawn_r = 3
    max_dist = 0.0

    Peter = DLA3DTree([0.0, 0.0, 0.0])

    for j in tqdm(range(100)):
        Brownian_array = [Brown3D([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], rand_len=512, r=.08) for i in
                          range(10)]

        for i in Brownian_array:
            i.start_pos_sphere(max_dist)
            i.gen_rand()
            path = i.gen_path()
            print(i.d)
            # plt.scatter(i.path[0], i.path[1], s=40, alpha=0.5)
            indices = Peter.check_path(path)
            if indices > -1:
                Peter.add_seed(i.path[:, indices])

                # max_z = np.amax(George.seed_array, axis=0)[2]
                # origin[2] = max_z
                # final_coords[2] = max_z
                seed_l = np.linalg.norm(i.path[:, indices])
                max_dist = np.maximum(max_dist, np.linalg.norm(i.path[:, indices]))
                Peter.adjust_origin()

    colors = iter(cm.rainbow(np.linspace(0, 1, len(Peter.seed_array))))

    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(len(Peter.seed_array)):
        ax.scatter(Peter.seed_array[i][0], Peter.seed_array[i][1], s=30, color=next(colors), alpha=1)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    Peter.gen_link_graph('black')

    plt.show()

    with open('coords.txt', 'w') as f:
        f.write(str(np.shape(Peter.seed_array)[0]) + '\n')
        f.write(str(np.shape(Peter.link_tree)[0]))
        f.write('\n')
        for coord in Peter.seed_array:
            f.write(str(coord))
            f.write('\n')
        for link in Peter.link_tree:
            f.write(str(link))
            f.write('\n')
