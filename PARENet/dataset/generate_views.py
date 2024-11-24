import blenderproc as bproc
import numpy as np


bproc.init()
# bproc.renderer.enable_distance_output(False)
bproc.renderer.enable_depth_output(False)

# Create a simple object:
loaded_blend = bproc.loader.load_blend("/data/cgaxis_antique_globe_65_07_blender/cgaxis_antique_globe_65_07_blender.blend", obj_types = ['mesh', 'light', 'camera'])

objects = []
for obj in loaded_blend:
    if isinstance(obj, bproc.types.MeshObject):
        print(f"MeshObject Found: {obj.get_name()}")
        objects.append(obj)

bproc.camera.set_resolution(1200, 1200)

start = 0
end = 30
step = 5

for i in range(start, end, step):
    print(f"Rendering at {i} degrees")
    cam_pose = bproc.math.build_transformation_mat([0, -1.3, 1], [np.pi / 3, 0, 0])
    bproc.camera.add_camera_pose(cam_pose, i // step)
    for obj in objects:
        obj.set_rotation_mat(bproc.math.build_transformation_mat(obj.get_location(), obj.get_rotation_euler() + [0, 0, np.radians(step)])[:3, :3], i // step)
    data = bproc.renderer.render()
    bproc.writer.write_hdf5("output/", data)

# Render the scene
# data = bproc.renderer.render()

# Write the rendering into an hdf5 file
# bproc.writer.write_hdf5("output/", data)