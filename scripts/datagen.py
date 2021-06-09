import os
import h5py
import math
import random
import numpy as np
from PIL import Image
from volpy import *

# SETTINGS
N_IMAGES = 100
N_SAMPLES_TARGET = 4096
WIDTH = 1024
HEIGHT = 1024
H5_NAME = 'test'
ENVPATH = '/home/niko/render-data/envmaps'
VOLPATH = '/home/niko/render-data/volumetric'

# init renderer
# Renderer.init(w = WIDTH, h = HEIGHT, vsync=False, pinned=True, visible=False)
Renderer.resize(WIDTH, HEIGHT)
Renderer.tonemap = False
Renderer.draw()
# random.seed(42)

# collect envmaps and volumes
def glob_directory(root, ext='.hdr'):
    entries = []
    for dirname, subdirs, files in os.walk(root):
        for fname in files:
            if os.path.splitext(fname)[1] == ext:
                entries.append(os.path.join(dirname, fname))
    return sorted(entries)

envmaps = glob_directory(ENVPATH, '.hdr')
volumes = glob_directory(VOLPATH, '.vdb')

# init h5 datasets
filename_input = H5_NAME + '_input.h5'
filename_target = H5_NAME + '_target.h5'
if os.path.isfile(filename_input): os.remove(filename_input)
if os.path.isfile(filename_target): os.remove(filename_target)
file_input = h5py.File(filename_input, 'w')
datasets_input = []
datasets_input.append(file_input.create_dataset('color', shape=(N_IMAGES, 3, HEIGHT, WIDTH), dtype=np.float16))
datasets_input.append(file_input.create_dataset('feature1', shape=(N_IMAGES, 3, HEIGHT, WIDTH), dtype=np.float16))
datasets_input.append(file_input.create_dataset('feature2', shape=(N_IMAGES, 3, HEIGHT, WIDTH), dtype=np.float16))
datasets_input.append(file_input.create_dataset('feature3', shape=(N_IMAGES, 3, HEIGHT, WIDTH), dtype=np.float16))
datasets_input.append(file_input.create_dataset('feature4', shape=(N_IMAGES, 3, HEIGHT, WIDTH), dtype=np.float16))
file_target = h5py.File(filename_target, 'w')
dataset_target = file_target.create_dataset('color', shape=(N_IMAGES, 3, HEIGHT, WIDTH), dtype=np.float16)

def uniform_sample_sphere():
    z = 1.0 - 2.0 * random.random();
    r = math.sqrt(max(0.0, 1.0 - z * z));
    phi = 2.0 * math.pi * random.random();
    return vec3(r * math.cos(phi), r * math.sin(phi), z);

def randomize_parameters():
    params = {}
    params['samples'] = random.randint(1, 9)
    params['max_bounces'] = random.randint(1, 128) 
    params['seed_input'] = random.randint(0, 2**31) 
    params['seed_target'] = random.randint(0, 2**31) 
    params['environment'] = Environment(random.choice(envmaps))
    params['environment'].strength = 0.5 + random.random() * 10
    params['show_environment'] = random.random() < 0.05
    params['volume'] = Volume(random.choice(volumes))
    params['volume'].albedo = vec3(random.random(), random.random(), random.random())
    params['volume'].phase = -0.9 + (random.random() * 1.8)
    params['volume'].density_scale = 0.1 + random.random() * 10
    bb_min, bb_max = params['volume'].AABB()
    center = bb_min + (bb_max - bb_min) * 0.5
    radius = (bb_max - center).length()
    params['cam_pos'] = center + uniform_sample_sphere() * radius
    target = center + uniform_sample_sphere() * radius * 0.1
    params['cam_dir'] = (target - params['cam_pos']).normalize()
    params['cam_fov'] = 25 + (random.random() * 70)
    return params

for i, params in enumerate([randomize_parameters() for i in range(N_IMAGES)]):
    print(f'rendering {i+1}/{N_IMAGES}..')
    Renderer.volume = params['volume']
    Renderer.environment = params['environment']
    Renderer.show_environment = params['show_environment']
    Renderer.bounces = params['max_bounces']
    Renderer.cam_pos = params['cam_pos'] 
    Renderer.cam_dir = params['cam_dir']
    Renderer.cam_fov = params['cam_fov']
    # render noisy
    Renderer.seed = params['seed_input']
    Renderer.render(params['samples'])
    for f, dataset in enumerate(datasets_input):
        data = np.flip(np.array(Renderer.fbo_data(f)), axis=0)
        dataset[i] = np.transpose(data.astype(np.float16), [2, 1, 0])
    # render converged
    Renderer.seed = params['seed_target']
    Renderer.render(N_SAMPLES_TARGET)
    data_target = np.flip(np.array(Renderer.fbo_data(0)), axis=0)
    dataset_target[i] = np.transpose(data_target.astype(np.float16), [2, 1, 0])

Renderer.shutdown()
