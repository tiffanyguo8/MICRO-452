"""Cinématique du Thymio (robot différentiel) et intégration de la pose.

Le Thymio est un robot à deux roues motrices indépendantes. On le modélise par
un modèle unicycle ``(v, omega)`` :

* ``v``     : vitesse linéaire du centre des roues, en m/s ;
* ``omega`` : vitesse angulaire, en rad/s, positive en tournant vers la gauche.

Les valeurs par défaut correspondent au Thymio II. Vérifiez-les sur votre robot
avant de les utiliser en boucle fermée sur le matériel réel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Pose


@dataclass(frozen=True)
class DifferentialDrive:
    """Géométrie et limites d'un robot différentiel."""

    wheel_radius: float = 0.0215
    """Rayon d'une roue, en mètres (Thymio II : ~21,5 mm)."""

    wheelbase: float = 0.095
    """Entraxe des deux roues, en mètres (Thymio II : ~95 mm)."""

    max_wheel_speed: float = 0.20
    """Vitesse linéaire maximale d'une roue, en m/s."""

    @property
    def max_linear_speed(self) -> float:
        """Vitesse linéaire maximale du robot (roues à vitesse max, tout droit)."""
        return self.max_wheel_speed

    @property
    def max_angular_speed(self) -> float:
        """Vitesse angulaire maximale (roues en sens opposés)."""
        return 2.0 * self.max_wheel_speed / self.wheelbase

    def wheel_speeds(self, v: float, omega: float) -> tuple[float, float]:
        """Convertit un twist ``(v, omega)`` en vitesses linéaires de roues.

        Returns
        -------
        (left, right)
            Vitesses linéaires au sol des roues gauche et droite, en m/s.
        """
        half = omega * self.wheelbase / 2.0
        return v - half, v + half

    def twist(self, left: float, right: float) -> tuple[float, float]:
        """Convertit des vitesses de roues en twist ``(v, omega)``."""
        v = (left + right) / 2.0
        omega = (right - left) / self.wheelbase
        return v, omega

    def wheel_angular_speeds(self, v: float, omega: float) -> tuple[float, float]:
        """Vitesses angulaires des roues, en rad/s (pour un encodeur)."""
        left, right = self.wheel_speeds(v, omega)
        return left / self.wheel_radius, right / self.wheel_radius

    def clamp_twist(self, v: float, omega: float) -> tuple[float, float]:
        """Réduit homothétiquement le twist pour respecter la vitesse des roues.

        On conserve le rapport ``omega / v`` (donc la courbure) et on réduit les
        deux composantes du même facteur : le robot ralentit dans les virages
        serrés au lieu de dévier de sa trajectoire.
        """
        left, right = self.wheel_speeds(v, omega)
        peak = max(abs(left), abs(right))
        if peak <= self.max_wheel_speed or peak < 1e-12:
            return v, omega
        scale = self.max_wheel_speed / peak
        return v * scale, omega * scale

    def aseba_speed(self, wheel_speed: float, *, max_unit: int = 500) -> int:
        """Convertit une vitesse de roue (m/s) en unité moteur Aseba.

        Le firmware Thymio attend une consigne entière dans ``[-500, 500]``.
        La correspondance exacte dépend de la charge et du sol : à calibrer.
        """
        ratio = wheel_speed / self.max_wheel_speed
        return int(round(max(-1.0, min(1.0, ratio)) * max_unit))


def integrate_unicycle(pose: Pose, v: float, omega: float, dt: float) -> Pose:
    """Intègre exactement le modèle unicycle sur ``dt`` secondes.

    Contrairement à une intégration d'Euler, le déplacement suit l'arc de
    cercle réel lorsque ``omega`` est non nul, ce qui reste précis même avec un
    pas de temps grossier.
    """
    x, y, theta = pose
    if abs(omega) < 1e-9:
        return (
            x + v * math.cos(theta) * dt,
            y + v * math.sin(theta) * dt,
            theta,
        )
    radius = v / omega
    theta_next = theta + omega * dt
    return (
        x + radius * (math.sin(theta_next) - math.sin(theta)),
        y - radius * (math.cos(theta_next) - math.cos(theta)),
        theta_next,
    )
