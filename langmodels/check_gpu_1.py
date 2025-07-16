import tensorflow as tf

def get_physical_devices():
    devs = tf.config.list_physical_devices()
    if devs:
        print("[0] TensorFlow is detecting the following device(s):")
        for dev in devs:
            print(f"    {dev}")
    else:
        print("[0] TensorFlow is not detecting any devices.")

    # List all physical devices visible to TensorFlow
    cpus = tf.config.list_physical_devices('CPU')
    if cpus:
        print("[1] TensorFlow is detecting the following CPU(s):")
        for cpu in cpus:
            print(f"    {cpu}")
    else:
        print("[1] TensorFlow is not detecting any CPU devices. Running on CPU.")

    # List all physical devices visible to TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("[2] TensorFlow is detecting the following GPU(s):")
        for gpu in gpus:
            print(f"    {gpu}")
    else:
        print("[2] TensorFlow is not detecting any GPU devices. Running on CPU.")

def get_logical_devices():
    devs = tf.config.list_logical_devices()
    if devs:
        print("[0] TensorFlow is detecting the following device(s):")
        for dev in devs:
            print(f"    {dev}")
    else:
        print("[0] TensorFlow is not detecting any devices.")

    # List all logical devices visible to TensorFlow
    cpus = tf.config.list_logical_devices('CPU')
    if cpus:
        print("[1] TensorFlow is detecting the following CPU(s):")
        for cpu in cpus:
            print(f"    {cpu}")
    else:
        print("[1] TensorFlow is not detecting any CPU devices. Running on CPU.")

    # List all logical devices visible to TensorFlow
    gpus = tf.config.list_logical_devices('GPU')
    if gpus:
        print("[2] TensorFlow is detecting the following GPU(s):")
        for gpu in gpus:
            print(f"    {gpu}")
    else:
        print("[2] TensorFlow is not detecting any GPU devices. Running on CPU.")


def get_cpu_devices():
    cpus = tf.config.list_physical_devices('CPU')
    if cpus:
        # Restrict TensorFlow to only use the first CPU
        try:
            tf.config.set_visible_devices(cpus[0], 'CPU')
            logical_cpus = tf.config.list_logical_devices('CPU')
            print("[3]", len(cpus), "Physical CPUs,", len(logical_cpus), "Logical CPU")
        except RuntimeError as e:
            # Visible devices must be set before CPUs have been initialized
            print(f"[3] {e}")

def get_gpu_devices():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print("[4]", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(f"[4] {e}")


print("Using TensorFlow to detect compute device(s):")
get_physical_devices()
get_logical_devices()
print(".....Completed TensorFlow proccesses.")

get_cpu_devices()
get_gpu_devices()

gpu_device_name = tf.test.gpu_device_name()
if gpu_device_name:
    print(f"[5] TensorFlow is configured to use the GPU device: {gpu_device_name}")
else:
    print("[5] TensorFlow is not configured to use a GPU device. Running on CPU.")
