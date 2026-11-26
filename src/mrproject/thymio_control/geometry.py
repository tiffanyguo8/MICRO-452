"""Primitives géométriques partagées par la librairie de contrôle.

Convention de repère (repère cartésien classique) :

* ``x`` vers la droite, ``y`` vers le haut ;
* ``theta`` est l'angle du robot par rapport à l'axe ``x``, en radians,
  positif dans le sens trigonométrique (anti-horaire).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vec2 = NDArray[np.float64]
Pose = tuple[float, float, float]


def wrap_to_pi(angle: float) -> float:
    """Ramène un angle dans l'intervalle ``]-pi, pi]``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wrap_to_pi_array(angles: ArrayLike) -> NDArray[np.float64]:
    """Version vectorisée de :func:`wrap_to_pi`."""
    values = np.asarray(angles, dtype=float)
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def rotation_matrix(theta: float) -> NDArray[np.float64]:
    """Matrice de rotation 2D d'angle ``theta``."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def to_local_frame(point: ArrayLike, pose: Pose) -> Vec2:
    """Exprime un point du repère monde dans le repère du robot."""
    x, y, theta = pose
    delta = np.asarray(point, dtype=float) - np.array([x, y], dtype=float)
    return rotation_matrix(theta).T @ delta


def to_world_frame(point: ArrayLike, pose: Pose) -> Vec2:
    """Exprime un point du repère robot dans le repère monde."""
    x, y, theta = pose
    return rotation_matrix(theta) @ np.asarray(point, dtype=float) + np.array(
        [x, y], dtype=float
    )


def distance(a: ArrayLike, b: ArrayLike) -> float:
    """Distance euclidienne entre deux points 2D."""
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def normalize(vector: ArrayLike) -> Vec2:
    """Renvoie le vecteur unitaire, ou le vecteur nul s'il est dégénéré."""
    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.zeros(2, dtype=float)
    return v / norm
