"""``env.scene`` provides scenarios, or the underlying environment in which the satellite can collect data.

Scenarios typically correspond to certain type(s) of :ref:`env.data` systems. The
following scenarios have been implemented:

* :class:`UniformTargets`: Uniformly distributed targets to be imaged by an :class:`~env.sats.ImagingSatellite`.
* :class:`CityTargets`: Targets distributed near population centers.
* :class:`UniformNadirScanning`: Uniformly desireable data over the surface of the Earth.
"""

from env.scene.scenario import Scenario, UniformNadirScanning
from env.scene.targets import CityTargets, UniformTargets

__doc_title__ = "Scenario"
__all__ = [
    "Scenario",
    "UniformTargets",
    "CityTargets",
    "UniformNadirScanning",
]
