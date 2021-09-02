from volpy import *

# TODO static instance or something

r = Renderer()
r.init()
r.tonemapping = True
r.show_environment = True
r.volume = Volume('data/head_8bit.dat')
r.volume.density_scale = 0.5
r.volume.albedo = vec3(1, 1, 1)
r.environment = Environment('data/clearsky.hdr')
r.render(128)
r.save('test.jpg')
