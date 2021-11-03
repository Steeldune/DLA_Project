import bpy
import numpy as np
from pathlib import Path

def gen_branch(sphere_rad, cyl_rad, loc1, loc2):

# add sphere1
    bpy.ops.surface.primitive_nurbs_surface_sphere_add(radius=sphere_rad, location=loc1)

    # add sphere2
    bpy.ops.surface.primitive_nurbs_surface_sphere_add(radius=sphere_rad, location=loc2)

    # add a curve to link them together
    bpy.ops.curve.primitive_bezier_curve_add()
    obj = bpy.context.object
    obj.data.dimensions = '3D'
    obj.data.fill_mode = 'FULL'
    obj.data.bevel_depth = cyl_rad
    obj.data.bevel_resolution = 4
    # set first point to centre of sphere1
    obj.data.splines[0].bezier_points[0].co = loc1
    obj.data.splines[0].bezier_points[0].handle_left_type = 'VECTOR'
    obj.data.splines[0].bezier_points[0].handle_right_type = 'VECTOR'
    # set second point to centre of sphere2
    obj.data.splines[0].bezier_points[1].co = loc2
    obj.data.splines[0].bezier_points[1].handle_left_type = 'VECTOR'
    obj.data.splines[0].bezier_points[1].handle_right_type = 'VECTOR'
    return True

coords_path = 'C:/Users/night_3ns60sk/OneDrive/Documenten/TU_algemeen/DLA_project/coords.txt'
f = Path(bpy.path.abspath(coords_path))

if f.exists():
    data = f.read_text()

data_split = data.split('\n')
data_len = int(data_split[0])
tree_len = int(data_split[1])
print(data_len)
print(tree_len)

particle_coords = np.zeros((data_len, 3), dtype=float)
tree_links = np.zeros((tree_len, 2), dtype=int)

for i in range(2, data_len+2):
    temp = data_split[i][1: -1]
    particle_coords[i-2] = np.fromstring(temp, dtype=float, count=3, sep=' ')

print('Done with getting particles')

for j in range(data_len+2, data_len+tree_len+2):
    temp_tree = data_split[j][1: -1]
    print(temp_tree)
    tree_links[j-data_len-2] = np.fromstring(temp_tree, dtype=int, count=2, sep=', ')
    
print('Done with getting links')
print(tree_links)

for i, link in enumerate(tree_links):
    gen_branch(0.05, 0.025, particle_coords[link[0]], particle_coords[link[1]])
    if i%100 ==0:
        print('Generating branch {}'.format(i))
