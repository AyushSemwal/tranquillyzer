"""
Module usage

First `import available_gpus` call sets up all functions within module and makes
them available to call. All imports of available_gpus will retain the same
values within a single run of tranquillyzer.

Detection vs. actual usage are reported separately:
- :func:`log_gpus_detected` reports what TensorFlow physically sees.
- :func:`log_gpus_in_use` reports how many / which of those the current command
  will actually run on (callers pass the resolved distribution strategy).
"""

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

# List of GPUs available to TensorFlow
# This is before any masking is done on GPUs to limit those available
_GPUS = tf.config.list_physical_devices("GPU")


def n_gpus():
    """Returns the total number of GPUs available to the user"""
    return len(_GPUS)


def get_tensorflow_output():
    """Returns raw list as given by tf.config.list_physical_devices('GPU')"""
    return _GPUS


def get_gpu_names_clean():
    """Returns a list of GPU names in the form 'GPU:i' where i is the index of
    the GPU. This should match well with the names TensorFlow returns without
    the extra information included.
    """
    return [f"GPU:{i}" for i, _ in enumerate(_GPUS)]


def get_gpu_names_raw():
    """Returns a list of the GPU names as retrieved from
    tf.config.list_physical_devices('GPU')
    """
    return [f"{dev.name}" for dev in _GPUS]


def gpus_to_visible_devices_string():
    """Provide the GPU numbers as a comma-separated string"""
    return ",".join(map(str, range(n_gpus())))


def log_gpus_detected():
    """Log how many physical GPUs TensorFlow sees. Makes no claim about usage."""
    if n_gpus() > 0:
        logger.info(f"GPUs detected: {n_gpus()} ({', '.join(get_gpu_names_clean())})")
    else:
        logger.info("No GPUs detected")


def log_gpus_in_use(strategy=None):
    """Log how many GPUs the current command will actually run on.

    Pass the distribution *strategy* the caller resolved (``MirroredStrategy``
    when training/inferring across multiple GPUs, ``None`` for single-device
    or CPU paths). The message distinguishes the multi-GPU strategy case from
    a single-device run, and from CPU-only mode.
    """
    names = get_gpu_names_clean()
    total = n_gpus()
    if total == 0:
        logger.info("Running in CPU-only mode")
        return
    if strategy is not None:
        n = int(getattr(strategy, "num_replicas_in_sync", total))
        used_names = names[:n] if n <= len(names) else names
        logger.info(f"Running on {n} GPU(s) via MirroredStrategy: {', '.join(used_names)}")
    else:
        logger.info(f"Running on 1 of {total} detected GPU(s): {names[0]}")


def available_gpus():
    """Formats a nice table for available GPUs"""
    if n_gpus() > 0:
        # Column names in table
        raw_header = "Raw Name"
        clean_header = "Clean Name"

        # Maximum widths to properly pad table
        max_raw_width = max([len(name) for name in get_gpu_names_raw()])
        if len(raw_header) > max_raw_width:
            max_raw_width = len(raw_header)

        max_clean_width = max([len(name) for name in get_gpu_names_clean()])
        if len(clean_header) > max_clean_width:
            max_clean_width = len(clean_header)

        print(f"{n_gpus()} GPUs found! Tranquillyzer is able to run in GPU mode")
        print("\n|", "-" * max_raw_width, "|", "-" * max_clean_width, "|", sep="-")
        print(f"| {raw_header:<{max_raw_width}} | {clean_header:<{max_clean_width}} |")
        print("|", "-" * max_raw_width, "|", "-" * max_clean_width, "|", sep="-")
        for raw, clean in zip(get_gpu_names_raw(), get_gpu_names_clean()):
            print(f"| {raw:<{max_raw_width}} | {clean:<{max_clean_width}} |")
        print("|", "-" * max_raw_width, "|", "-" * max_clean_width, "|", sep="-")

    else:
        print("No GPUs found! Tranquillyzer will run in CPU-only mode.")
