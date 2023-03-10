import os
import sys
import numpy as np
from scipy.stats import qmc
# import colmap import/export script
sys.path.append(os.path.dirname(__file__))
import read_write_model as colmap
# import renderer module
import volpy

# helper functions
def sample_unit_sphere(sample):
    import math
    z = 1.0 - 2.0 * sample[0]
    r = math.sqrt(max(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * sample[1]
    return volpy.vec3(r * math.cos(phi), r * math.sin(phi), z)

if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

    # settings
    OUT_PATH = os.path.join(ROOT_DIR, 'colmap')
    OUT_FORMAT = ".txt"
    N_VIEWS = 256
    VOLUME = os.path.join(ROOT_DIR, 'data/smoke.brick')
    ALBEDO = volpy.vec3(0.9, 0.9, 0.9)
    PHASE = 0.5
    DENSITY_SCALE = 1.0
    ENVMAP = os.path.join(ROOT_DIR, 'data/table_mountain_2_puresky_1k.hdr')
    ENV_STRENGTH = 2.0
    SAMPLES = 1 << 12
    BOUNCES = 128
    FOVY = 70
    SEED = 42
    BACKGROUND = True
    TONEMAPPING = True

    print(OUT_PATH, VOLUME, ENVMAP)

    # ------------------------------------------
    # Render colmap dataset

    # init
    renderer = volpy.Renderer()
    renderer.init()
    renderer.draw()
    os.makedirs(OUT_PATH, exist_ok=True)

    # setup scene
    renderer.seed = SEED
    renderer.bounces = BOUNCES
    renderer.volume = volpy.Volume(VOLUME)
    renderer.albedo = ALBEDO
    renderer.phase = PHASE
    renderer.density_scale = DENSITY_SCALE
    renderer.environment = volpy.Environment(ENVMAP)
    renderer.environment.strength = ENV_STRENGTH
    renderer.show_environment = BACKGROUND
    renderer.tonemapping = TONEMAPPING
    renderer.scale_and_move_to_unit_cube()
    renderer.commit()

    cameras = {}
    images = {}
    points3D = {}

    # HACK: write world-space AABB of volume as point3D (pos + rgb) to dataset
    points3D[0] = colmap.Point3D(id=0, xyz=np.array(renderer.volume.AABB("density")[0]), rgb=np.array(renderer.volume.AABB("density")[1]), error=0, image_ids=np.array([]), point2D_idxs=np.array([]))

    # write camera
    cameras[0] = colmap.Camera(id=0, model="SIMPLE_PINHOLE", width=renderer.resolution().x, height=renderer.resolution().y, params=np.array([renderer.colmap_focal_length(), renderer.resolution().x//2, renderer.resolution().y//2]))

    # random sampler
    samplerOut = qmc.Sobol(d=2, seed=SEED+1)
    samplerIn = qmc.Sobol(d=2, seed=SEED+2)

    # write views
    for i in range(N_VIEWS):
        print(f'rendering {i+1}/{N_VIEWS}..')
        # setup camera
        bb_min, bb_max = renderer.volume.AABB("density")
        center = bb_min + (bb_max - bb_min) * 0.5
        radius = (bb_max - center).length()
        renderer.cam_pos = center + sample_unit_sphere(samplerOut.random()[0, 0:2]) * radius
        renderer.cam_dir = (center + sample_unit_sphere(samplerIn.random()[0, 0:2]) * radius * 0.1 - renderer.cam_pos).normalize()
        renderer.cam_fov = FOVY
        # render view
        renderer.render(SAMPLES)
        renderer.draw()
        # store view
        filename = f"view_{i:06}.png"
        renderer.save_with_alpha(os.path.join(OUT_PATH, filename))
        images[i] = colmap.Image(id=i, qvec=np.array(renderer.colmap_view_rot())[[3, 0, 1, 2]], tvec=np.array(renderer.colmap_view_trans()), camera_id=0, name=filename, xys=np.array([]), point3D_ids=np.array([]))

    print('--------------------')
    print("#cameras:", len(cameras))
    print("#images:", len(images))
    print("#points3D:", len(points3D))

    colmap.write_model(cameras, images, points3D, path=OUT_PATH, ext=OUT_FORMAT)

    renderer.shutdown()
