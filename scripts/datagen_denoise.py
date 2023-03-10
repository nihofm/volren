import os
import h5py
import math
import random
import numpy as np
# import renderer module
import volpy

if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

    # settings
    N_IMAGES = 256
    N_SAMPLES_TARGET = 1 << 12
    SEED = 42
    H5_NAME = 'dataset'
    VOLPATH = os.path.join(ROOT_DIR, './data')
    ENVPATH = os.path.join(ROOT_DIR, './data')
    ENABLE_RANDOM_TRANSFERFUNC = False

    # init renderer
    renderer = volpy.Renderer()
    renderer.init()
    renderer.draw()
    random.seed(SEED)

    # collect envmap and volume files recursively
    def glob_directory(root, ext='.hdr'):
        entries = []
        for dirname, _, files in os.walk(root):
            for fname in files:
                if os.path.splitext(fname)[1] == ext:
                    entries.append(os.path.join(dirname, fname))
        return sorted(entries)

    volumes = glob_directory(VOLPATH, '.brick')
    envmaps = glob_directory(ENVPATH, '.hdr')

    print('#volumes:', len(volumes))
    print('#envmaps:', len(envmaps))

    # init h5 datasets
    SIZE = renderer.resolution()
    filename_input = H5_NAME + '_input.h5'
    filename_target = H5_NAME + '_target.h5'
    if os.path.isfile(filename_input): os.remove(filename_input)
    if os.path.isfile(filename_target): os.remove(filename_target)
    file_input = h5py.File(filename_input, 'w')
    dataset_input = file_input.create_dataset('color', shape=(N_IMAGES, 3, SIZE.y, SIZE.x), dtype=np.float16)
    file_target = h5py.File(filename_target, 'w')
    dataset_target = file_target.create_dataset('color', shape=(N_IMAGES, 3, SIZE.y, SIZE.x), dtype=np.float16)

    def uniform_sample_sphere():
        z = 1.0 - 2.0 * random.random()
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = 2.0 * math.pi * random.random()
        return volpy.vec3(r * math.cos(phi), r * math.sin(phi), z)

    def randomize_parameters():
        params = {}
        # define range of randomized parameters
        params['samples'] = random.randint(1, 32+1)
        params['max_bounces'] = random.randint(1, 128+1)
        params['seed_input'] = random.randint(0, 2**31)
        params['seed_target'] = random.randint(0, 2**31)
        params['env_path'] = random.choice(envmaps)
        params['env_strength'] = 0.5 + random.random() * 10
        params['env_show'] = random.random() < 0.1
        params['lut_n_bins'] = random.randint(2, 32+1)
        params['lut_window_left'] = random.random() * 0.25
        params['lut_window_width'] = random.random()
        params['vol_path'] = random.choice(volumes)
        params['vol_albedo'] = volpy.vec3(random.random(), random.random(), random.random())
        params['vol_phase'] = -0.9 + (random.random() * 1.8)
        params['vol_density_scale'] = 0.01 + random.random() * 5
        params['cam_pos_sample'] = uniform_sample_sphere()
        params['cam_dir_sample'] = uniform_sample_sphere()
        params['cam_fov'] = 25 + (random.random() * 70)
        return params

    for i, params in enumerate([randomize_parameters() for i in range(N_IMAGES)]):
        print(f'rendering {i+1}/{N_IMAGES}..')
        # load volume
        renderer.volume = volpy.Volume(params['vol_path'])
        renderer.commit()
        renderer.albedo = params['vol_albedo']
        renderer.phase = params['vol_phase']
        renderer.density_scale = params['vol_density_scale']
        # load envmap
        renderer.environment = volpy.Environment(params['env_path'])
        renderer.environment.strength = params['env_strength']
        renderer.show_environment = params['env_show']
        # randomze transferfunc?
        if (ENABLE_RANDOM_TRANSFERFUNC):
            renderer.transferfunc = volpy.TransferFunction()
            renderer.transferfunc.randomize(params['lut_n_bins'])
            renderer.transferfunc.window_left = params['lut_window_left']
            renderer.transferfunc.window_width = params['lut_window_width']
        else:
            renderer.transferfunc = None
        # setup camera
        bb_min, bb_max = renderer.volume.AABB("density")
        center = bb_min + (bb_max - bb_min) * 0.5
        radius = (bb_max - center).length()
        renderer.cam_pos = center + params['cam_pos_sample'] * radius
        renderer.cam_dir = (center + params['cam_dir_sample'] * radius * 0.1 - renderer.cam_pos).normalize()
        renderer.cam_fov = params['cam_fov']
        # render noisy
        renderer.seed = params['seed_input']
        renderer.bounces = params['max_bounces']
        renderer.render(params['samples'])
        data_input = np.flip(np.array(renderer.fbo_data()), axis=0)
        dataset_input[i] = np.transpose(data_input.astype(np.float16), [2, 1, 0])
        renderer.draw()
        # render converged reference
        renderer.seed = params['seed_target']
        renderer.bounces = params['max_bounces']
        renderer.render(N_SAMPLES_TARGET)
        data_target = np.flip(np.array(renderer.fbo_data()), axis=0)
        dataset_target[i] = np.transpose(data_target.astype(np.float16), [2, 1, 0])
        renderer.draw()

    renderer.shutdown()
