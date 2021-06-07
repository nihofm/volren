from volpy import *

# Renderer.init(w = 1024, h = 1024, vsync=False, pinned=True, visible=True)
Renderer.tonemapping = False
Renderer.show_environment = True
Renderer.volume = Volume('data/head_8bit.dat')
Renderer.volume.density_scale = 0.5
Renderer.volume.albedo = vec3(1, 1, 1)
Renderer.environment = Environment('data/clearsky.hdr')
Renderer.commit()
