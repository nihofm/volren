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
LUTPATH = '/home/niko/dev/volren/data'

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
# luts = glob_directory(LUTPATH, '.txt')

# init h5 datasets
filename_input = H5_NAME + '_input.h5'
filename_target = H5_NAME + '_target.h5'
if os.path.isfile(filename_input): os.remove(filename_input)
if os.path.isfile(filename_target): os.remove(filename_target)
file_input = h5py.File(filename_input, 'w')
datasets_input = []
for i in range(Renderer.fbo_num_buffers()):
    if i == 0:
        datasets_input.append(file_input.create_dataset('color', shape=(N_IMAGES, 3, HEIGHT, WIDTH), dtype=np.float16))
    else:
        datasets_input.append(file_input.create_dataset(f'feature{i}', shape=(N_IMAGES, 3, HEIGHT, WIDTH), dtype=np.float16))
file_target = h5py.File(filename_target, 'w')
dataset_target = file_target.create_dataset('color', shape=(N_IMAGES, 3, HEIGHT, WIDTH), dtype=np.float16)

def uniform_sample_sphere():
    z = 1.0 - 2.0 * random.random();
    r = math.sqrt(max(0.0, 1.0 - z * z));
    phi = 2.0 * math.pi * random.random();
    return vec3(r * math.cos(phi), r * math.sin(phi), z);

def randomize_parameters():
    params = {}
    params['samples'] = random.randint(1, 8+1)
    params['max_bounces'] = random.randint(1, 128+1)
    params['seed_input'] = random.randint(0, 2**31) 
    params['seed_target'] = random.randint(0, 2**31) 
    params['env_path'] = random.choice(envmaps)
    params['env_strength'] = 0.5 + random.random() * 10
    params['env_show'] = random.random() < 0.05
    # params['tf_lo'] = vec4(random.random(), random.random(), random.random(), 0.0)
    # params['tf_hi'] = vec4(random.random(), random.random(), random.random(), 1.0)
    # params['tf_path'] = random.choice(luts)
    # params['tf_left'] = random.random() * 0.5
    # params['tf_width'] = 0.1 + random.random() * 0.9
    params['vol_path'] = random.choice(volumes)
    params['vol_albedo'] = vec3(random.random(), random.random(), random.random())
    params['vol_phase'] = -0.9 + (random.random() * 1.8)
    params['vol_scale'] = 0.1 + random.random() * 10
    params['cam_pos_sample'] = uniform_sample_sphere()
    params['cam_dir_sample'] = uniform_sample_sphere()
    params['cam_fov'] = 25 + (random.random() * 70)
    return params

for i, params in enumerate([randomize_parameters() for i in range(N_IMAGES)]):
    print(f'rendering {i+1}/{N_IMAGES}..')
    # load volume
    Renderer.volume = Volume(params['vol_path'])
    Renderer.volume.albedo = params['vol_albedo']
    Renderer.volume.phase = params['vol_phase']
    Renderer.volume.density_scale = params['vol_scale']
    # load envmap
    Renderer.environment = Environment(params['env_path'])
    Renderer.environment.strength = params['env_strength']
    Renderer.show_environment = params['env_show']
    # load transferfunc
    # Renderer.transferfunc = TransferFunction(params['tf_path'])
    # Renderer.transferfunc = TransferFunction([params['tf_lo'], params['tf_hi']])
    # Renderer.transferfunc.window_left = params['tf_left']
    # Renderer.transferfunc.window_width = params['tf_width']
    # setup camera
    bb_min, bb_max = Renderer.volume.AABB()
    center = bb_min + (bb_max - bb_min) * 0.5
    radius = (bb_max - center).length()
    Renderer.cam_pos = center + params['cam_pos_sample'] * radius
    Renderer.cam_dir = (center + params['cam_dir_sample'] * radius * 0.1 - Renderer.cam_pos).normalize()
    Renderer.cam_fov = params['cam_fov']
    # render noisy
    Renderer.seed = params['seed_input']
    Renderer.bounces = params['max_bounces']
    Renderer.render(params['samples'])
    for f, dataset in enumerate(datasets_input):
        data = np.flip(np.array(Renderer.fbo_data(f)), axis=0)
        dataset[i] = np.transpose(data.astype(np.float16), [2, 1, 0])
    # render converged reference
    Renderer.seed = params['seed_target']
    Renderer.bounces = 128
    Renderer.render(N_SAMPLES_TARGET)
    data_target = np.flip(np.array(Renderer.fbo_data(0)), axis=0)
    dataset_target[i] = np.transpose(data_target.astype(np.float16), [2, 1, 0])

Renderer.shutdown()
