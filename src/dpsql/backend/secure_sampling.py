import logging
import random
from enum import Enum, auto

import opendp.prelude as dp
from opendp.domains import atom_domain
from opendp.metrics import absolute_distance
from opendp.mod import enable_features

enable_features("contrib")


logger = logging.getLogger(__name__)


class SamplingMode(Enum):
    SECURE = auto()  # Use OpenDP for secure Gaussian sampling (default)
    DEBUG = auto()  # Use fixed seed for reproducible results
    AUDIT = auto()  # Use random module for faster auditing


SAMPLING_MODE = SamplingMode.SECURE
INPUT_SPACE = atom_domain(T=float, nan=False), absolute_distance(T=float)


def secure_gauss(input: float, sigma: float) -> float:
    """
    Add Gaussian noise to an input value using OpenDP.

    Args:
        input (float): The input value to add noise to.
        sigma (float): The standard deviation of the Gaussian noise.

    Returns:
        float: The input value with Gaussian noise added.
    """

    logger.debug(
        "secure_gauss: mode=%s input=%s sigma=%s", SAMPLING_MODE.name, input, sigma
    )
    if SAMPLING_MODE == SamplingMode.SECURE:
        gauss = dp.m.make_gaussian(*INPUT_SPACE, scale=sigma)
        val = gauss(input)
        logger.debug("secure_gauss (OpenDP): result=%s", val)
        return val
    elif SAMPLING_MODE == SamplingMode.DEBUG:
        random.seed(0)
        val = random.gauss(input, sigma)
        logger.debug("secure_gauss (DEBUG): result=%s", val)
        return val
    elif SAMPLING_MODE == SamplingMode.AUDIT:
        val = random.gauss(input, sigma)
        logger.debug("secure_gauss (AUDIT): result=%s", val)
        return val
    else:
        logger.error("Invalid sampling mode: %s", SAMPLING_MODE)
        raise ValueError("Invalid sampling mode.")
