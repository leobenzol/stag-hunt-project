import os

# Force TensorFlow to run on the CPU only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Reproducibility: make TF ops deterministic so a given (seed, hyperparameters) run is
# byte-reproducible. Without this, CPU reduction/thread scheduling nondeterminism tips the
# near-Bernoulli stag-hunt "escape to the cooperative equilibrium" differently on identical
# reruns, which makes any convergence tuning impossible to lock in. Single-threaded
# (set by the launcher) + fixed seeds + op determinism => identical results across reruns.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
# oneDNN reorders float reductions across runs ("slightly different numerical results due to
# floating-point round-off from different computation orders"), which is enough to flip a
# near-Bernoulli escape between reruns. Disable it so results are byte-reproducible.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    import tensorflow as tf  # noqa: E402

    tf.config.experimental.enable_op_determinism()
except Exception:  # pragma: no cover - TF not always importable at this point
    pass
