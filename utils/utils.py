import os, pdb
import numpy as np

def normalize_vertex(vertex):
    '''
    normalize vertex to -1 1, input vertex type is numpy array
    '''
    norm_vertex = 2 * (vertex - np.min(vertex)) / (np.max(vertex) - np.min(vertex)) + (-1)
    return norm_vertex

def get_obj_vertex_ali(file):
    '''
    get obj vertex, some obj file can not be loaded through open3d or trimesh
    '''
    with open(file, 'r') as f:
        vertex_group = []
        part_vertex = []
        last_fisrt = ''

        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            split_line = line.split(' ')
            curr_first = split_line[0]
            if curr_first != last_fisrt:
                if part_vertex != []: vertex_group += part_vertex
                part_vertex = []
            if 'v' == curr_first:
                try:
                    vertex = [float(split_line[-3]), float(split_line[-2]), float(split_line[-1])]
                except:
                    continue
                    pdb.set_trace()
                    vertex = [float(split_line[-3]), float(split_line[-2]), float(split_line[-1])]
                part_vertex.append(vertex)

            last_fisrt = curr_first

        # remove shadow vertex
        if len(vertex_group[-1]) == 4 and len(vertex_group) != 0:
            vertex_group.pop()
    
    return np.array(vertex_group)

def get_obj_vertex_open3d(file):
    obj_file = os.path.join(model_dir, model_id + '.obj')
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh_vertices = np.asarray(mesh.vertices)
    return mesh_vertices        

def replace_and_save_obj(input_vertex, raw_obj_file, save_file):
    all_input_vertex = []
    for i, input_part_vertex in enumerate(input_vertex):
        all_input_vertex.append(input_part_vertex)
     
    valid_lines = []
    with open(raw_obj_file, 'r') as f:

        v_id = 0
        lines = f.readlines()
        try:
            for i, line in enumerate(lines):
                split_line = line.split(' ')
                curr_first = split_line[0]

                # check if shadow exist in line
                for element in split_line:
                    if 'shadow' in element:
                        break
                if 'shadow' in split_line[-1] or 'Shadow' in split_line[-1] or 'SHADOW' in split_line[-1]:
                    break

                if 'v' == curr_first:
                    curr_vertex = all_input_vertex[v_id]
                    curr_line = 'v  ' + str(curr_vertex[0]) + ' ' + str(curr_vertex[1]) + ' ' + str(curr_vertex[2]) + '\n'
                    v_id += 1
                else:
                    curr_line = line
                valid_lines.append(curr_line)
            f.close()
        except:
            print(raw_obj_file)
    
    if len(valid_lines) < 50:
        return None
    
    with open(save_file, 'w') as f:
        for valid_line in valid_lines:
            f.write(valid_line)
        f.close()


