import volpy

path = '/home/niko/render-data/volumetric/bunny_cloud.vdb'
grid = 'density'
print(f'loading {path} / {grid}...')
g = volpy.Grid(path, grid)
print(f'grid name: {g.gridName()}, active voxels: {g.activeVoxelCount()}')
print(f'world AABB: {g.worldBBoxMin()} / {g.worldBBoxMax()}')
print(f'index AABB: {g.indexBBoxMin()} / {g.indexBBoxMax()}')
print(f'voxel size: {g.voxelSize()}')

print('---')

def test_operators(v):
    print(v)
    v += v
    print(v + v)
    v *= v
    print(v * v)
    v /= v
    print(v / v)

v = volpy.vec3(10)
test_operators(v)
v = volpy.ivec3(10)
test_operators(v)
v = volpy.uvec3(10)
test_operators(v)
