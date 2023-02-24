# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner

# Optimize camera pose looking at shadow

# Use GPU if available
pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

# Set up the scene
with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    position = tf.Variable([0.0, 9.0, 0.0], dtype=tf.float32)
    look_at = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32)
    up = tf.Variable([0.0, 0.0, 1.0], dtype=tf.float32)
    fov = tf.Variable([45.0], dtype=tf.float32)
    clip_near = 1e-2
    resolution = (256, 256)
    cam = pyredner.Camera(position = position,
                          look_at = look_at,
                          up = up,
                          fov = fov,
                          clip_near = clip_near,
                          resolution = resolution)

with tf.device(pyredner.get_device_name()):
    mat_grey = pyredner.Material(
        diffuse_reflectance = tf.Variable([0.5, 0.5, 0.5], dtype=tf.float32))
    mat_black = pyredner.Material(
        diffuse_reflectance = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32))
    materials = [mat_grey, mat_black]

    # tf.constant allocates arrays on host memory for int32 arrays (some tensorflow internal mess),
    # but pyredner.Shape constructor automatically converts the memory to device if necessary.
    floor_vertices = tf.Variable([[-20.0,0.0,-20.0],[-20.0,0.0,20.0],[20.0,0.0,-20.0],[20.0,0.0,20.0]],
        dtype=tf.float32)
    floor_indices = tf.constant([[0,1,2], [1,3,2]], dtype=tf.int32)
    shape_floor = pyredner.Shape(floor_vertices, floor_indices, 0)
    blocker_vertices = tf.Variable([[-0.5,10.0,-0.5],[-0.5,10.0,0.5],[0.5,10.0,-0.5],[0.5,10.0,0.5]],
        dtype=tf.float32)
    blocker_indices = tf.constant([[0,1,2], [1,3,2]], dtype=tf.int32)
    shape_blocker = pyredner.Shape(blocker_vertices, blocker_indices, 0)
    light_vertices = tf.Variable([[-0.1,15,-0.1],[-0.1,15,0.1],[0.1,15,-0.1],[0.1,15,0.1]],
        dtype=tf.float32)
    light_indices = tf.constant([[0,2,1], [1,2,3]], dtype=tf.int32)
    shape_light = pyredner.Shape(light_vertices, light_indices, 1)
    shapes = [shape_floor, shape_blocker, shape_light]

with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    light_intensity = tf.Variable([5000.0, 5000.0, 5000.0], dtype=tf.float32)
# The first argument is the shape id of the light
light = pyredner.AreaLight(2, light_intensity)
area_lights = [light]
scene = pyredner.Scene(cam, shapes, materials, area_lights)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1)

# Alias of the render function

# Render our target
img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, 'results/test_shadow_camera/target.exr')
pyredner.imwrite(img, 'results/test_shadow_camera/target.png')
target = pyredner.imread('results/test_shadow_camera/target.exr')

# Perturb the scene, this is our initial guess
with tf.device(pyredner.get_device_name()):
    position = tf.Variable([-2.0, 7.0, 2.0], trainable=True)
scene.camera = pyredner.Camera(position = position,
                               look_at = look_at,
                               up = up,
                               fov = fov,
                               clip_near = clip_near,
                               resolution = resolution)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
# Render the initial guess
img = pyredner.render(1, *scene_args)
pyredner.imwrite(img, 'results/test_shadow_camera/init.png')
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_shadow_camera/init_diff.png')

# Optimize for blocker vertices
optimizer = tf.compat.v1.train.AdamOptimizer(5e-2)
for t in range(200):
    print('iteration:', t)
    
    with tf.GradientTape() as tape:
        # Forward pass: render the image
        # Need to rerun the Camera constructor for PyTorch autodiff to compute the derivatives
        scene.camera = pyredner.Camera(position   = position,
                                       look_at    = look_at,
                                       up         = up,
                                       fov        = fov,
                                       clip_near  = clip_near,
                                       resolution = resolution)
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4,
            max_bounces = 1)
        img = pyredner.render(t+1, *scene_args)
        pyredner.imwrite(img, 'results/test_shadow_camera/iter_{}.png'.format(t))
        loss = tf.reduce_sum(tf.square(img - target))
    print('loss:', loss)

    grads = tape.gradient(loss, [position])

    optimizer.apply_gradients(zip(grads, [position]))
    print('position.grad:', grads[0])
    print('position:', position)

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
img = pyredner.render(202, *scene_args)
pyredner.imwrite(img, 'results/test_shadow_camera/final.exr')
pyredner.imwrite(img, 'results/test_shadow_camera/final.png')
pyredner.imwrite(tf.abs(target - img), 'results/test_shadow_camera/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_shadow_camera/iter_%d.png", "-vb", "20M",
    "results/test_shadow_camera/out.mp4"])
