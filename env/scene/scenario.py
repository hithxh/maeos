"""Scenarios define data available for satellites to collect."""

import logging
from abc import ABC
from typing import TYPE_CHECKING

from env.utils.functional import Resetable

if TYPE_CHECKING:  # pragma: no cover
    from env.data.base import Data
    from env.sats import Satellite

logger = logging.getLogger(__name__)


class Scenario(ABC, Resetable):
    """Base scenario class."""

    def __init__(self) -> None:
        self.satellites: list["Satellite"]

    def link_satellites(self, satellites: list["Satellite"]) -> None:
        """Link the environment satellite list to the scenario.

        Args:
            satellites: List of satellites to communicate between.
        """
        self.satellites = satellites


class UniformNadirScanning(Scenario):
    """Defines a nadir target center at the center of the planet."""

    def __init__(self, value_per_second: float = 1.0) -> None:
        """Construct uniform data over the surface of the planet.

        Can be used with :class:`~env.data.ScanningTimeReward`.

        Args:
            value_per_second: Reward per second for imaging nadir.
        """
        self.value_per_second = value_per_second


__all__ = []
