"""Librairie de contrôle du robot éducatif Thymio.

Suivi d'une trajectoire pré-calculée (sortie d'A*) à partir de la seule pose
``(x, y, theta)`` fournie par la perception.

Le paquet ne dépend que de ``numpy`` : ni matplotlib, ni pytest, ni aucun code
de simulation. Il est donc directement embarquable sur le robot.

Exemple minimal
---------------
>>> from thymio_control import Trajectory, PurePursuitController
>>> path = Trajectory([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
>>> controller = PurePursuitController()
>>> current_state = (0.0, 0.0, 0.0) #x, y, theta
>>> command = controller.compute(current_state, path)
>>> round(command.v, 3) > 0
True
"""

from __future__ import annotations

from .controller import ControlCommand, PurePursuitConfig, PurePursuitController
from .geometry import (
    Pose,
    distance,
    normalize,
    rotation_matrix,
    to_local_frame,
    to_world_frame,
    wrap_to_pi,
    wrap_to_pi_array,
)
from .kinematics import DifferentialDrive, integrate_unicycle
from .trajectory import Projection, Trajectory

__all__ = [
    "ControlCommand",
    "DifferentialDrive",
    "Pose",
    "Projection",
    "PurePursuitConfig",
    "PurePursuitController",
    "Trajectory",
    "distance",
    "integrate_unicycle",
    "normalize",
    "rotation_matrix",
    "to_local_frame",
    "to_world_frame",
    "wrap_to_pi",
    "wrap_to_pi_array",
]

__version__ = "0.1.0"
