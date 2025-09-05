from __future__ import annotations

from typing import Any, Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from flax import struct
import yaml

from ..core.utils import yaml_safe

Array = jnp.ndarray
@struct.dataclass
class SamplerState:
    obs_params: Any
    chem_params: Any
    disk_params: Any
    base_disk: Any
    rays: Any
    mol: Any
    beam: Any     
    sigma: float = 1.0  # sigma can be scalar or an array matching the data shape
    adapter: Any = None 
    use_pressure_correction: bool = True


def append_inference_to_yaml(filepath, mcmc_params, results=None):
    """
    Append MCMC inference parameters and (optionally) results to an existing YAML file.

    Parameters
    ----------
    filepath : str
        Path to an existing YAML file created from disk/chemistry/observation params.
    mcmc_params : dict
        Dictionary of inference parameters.
        Example::

            {
                "method": "emcee",
                "numpy_seed": 123,
                "nwalkers": 64,
                "nburn": 2_000,
                "nsteps": 10_000,
                "sampled_params": ["v_turb"],
                "bounds": {"v_turb": [1e-3, 1e-1]}
            }

    results : dict, optional
        Dictionary of results to store.
        Example::

            {
                "v_turb_true": 0.01,
                "v_turb_map": 0.012,
                "v_turb_med": 0.011,
                "v_turb_ci68": [0.009, 0.013],
                "rms_map": 3.4e-3
            }

    Returns
    -------
    dict
        The merged YAML dictionary (with safe types).
    """
    # Load existing YAML
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}

    # Sanitize values
    mcmc_params = yaml_safe(mcmc_params)
    results = yaml_safe(results) if results is not None else None

    # Insert inference section
    config["inference"] = {"mcmc": mcmc_params}
    if results is not None:
        config["inference"]["results"] = results

    # Save back
    with open(filepath, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


