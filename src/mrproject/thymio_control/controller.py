"""Suivi de trajectoire par « pure pursuit ».

Principe : on projette la position du robot sur le chemin, on choisit un point
cible situé ``lookahead`` mètres plus loin le long de l'arc, puis on commande un
arc de cercle passant par ce point. La courbure nécessaire vaut

    kappa = 2 * y_local / d^2

où ``(x_local, y_local)`` est la cible exprimée dans le repère du robot et
``d`` sa distance. On en déduit ``omega = v * kappa``.

Le contrôleur ne dépend que de ``(x, y, theta)`` : il s'interface donc
directement avec la perception développée par le reste du groupe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Pose, Vec2, to_local_frame, wrap_to_pi
from .trajectory import Trajectory


@dataclass
class PurePursuitConfig:
    """Réglages du pure pursuit et du profil de vitesse."""

    min_lookahead: float = 0.12
    """Borne basse de la distance de visée adaptative."""

    max_lookahead: float = 0.45
    """Borne haute de la distance de visée adaptative."""

    lookahead_gain: float = 1.5
    """``lookahead = gain * v + min_lookahead``, en secondes."""

    max_linear_speed: float = 0.12
    """Vitesse linéaire de croisière, en m/s."""

    min_linear_speed: float = 0.02
    """Vitesse plancher pour ne pas caler en fin de trajectoire."""

    max_angular_speed: float = 3.0
    """Vitesse angulaire maximale, en rad/s."""

    slowdown_radius: float = 0.20
    """Distance à l'arrivée sous laquelle on décélère, en mètres."""

    curvature_slowdown: float = 0.6
    """Ralentissement en virage : ``v / (1 + k * |kappa|)``."""

    goal_tolerance: float = 0.03
    """Tolérance de position sur le point d'arrivée, en mètres."""

    heading_tolerance: float = 0.20
    """Tolérance de cap à l'arrivée, en radians."""

    projection_back: float = 0.30
    """Fenêtre de recherche arrière pour la projection, en mètres."""

    projection_ahead: float = 0.60
    """Fenêtre de recherche avant pour la projection, en mètres."""


@dataclass(frozen=True)
class ControlCommand:
    """Commande envoyée au robot pour un pas de temps."""

    v: float
    """Vitesse linéaire commandée, en m/s."""

    omega: float
    """Vitesse angulaire commandée, en rad/s."""

    s: float
    """Abscisse curviligne projetée du robot."""

    lookahead_point: Vec2
    """Point de visée effectivement utilisé."""

    lookahead: float
    """Distance de visée effective, en mètres."""

    curvature: float
    """Courbure commandée, en 1/m."""

    cross_track_error: float
    """Erreur latérale signée (positive à gauche du chemin), en mètres."""

    heading_error: float
    """Écart de cap robot / tangente, en radians."""

    remaining: float
    """Distance restante le long du chemin, en mètres."""

    finished: bool
    """Vrai lorsque le robot est arrivé et peut être arrêté."""

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "v": self.v,
            "omega": self.omega,
            "s": self.s,
            "curvature": self.curvature,
            "cross_track_error": self.cross_track_error,
            "heading_error": self.heading_error,
            "remaining": self.remaining,
            "finished": self.finished,
        }


@dataclass
class PurePursuitController:
    """Contrôleur de suivi de trajectoire sans état externe.

    Le contrôleur mémorise seulement la dernière vitesse commandée (pour
    adapter la distance de visée) et la dernière abscisse curviligne (pour
    restreindre la projection autour de la position courante).
    """

    config: PurePursuitConfig = field(default_factory=PurePursuitConfig)
    _last_v: float = 0.0
    _last_s: float | None = None

    def reset(self) -> None:
        """Réinitialise l'état interne (à appeler entre deux trajectoires)."""
        self._last_v = 0.0
        self._last_s = None

    def lookahead_distance(self, v: float | None = None) -> float:
        """Distance de visée adaptative, bornée par la configuration."""
        speed = self._last_v if v is None else v
        raw = self.config.lookahead_gain * abs(speed) + self.config.min_lookahead
        return float(
            np.clip(raw, self.config.min_lookahead, self.config.max_lookahead)
        )

    def compute(
        self,
        pose: Pose,
        trajectory: Trajectory,
        *,
        s_hint: float | None = None,
    ) -> ControlCommand:
        """Calcule la commande ``(v, omega)`` pour la pose courante."""
        cfg = self.config
        hint = self._last_s if s_hint is None else s_hint

        proj = trajectory.project(
            (pose[0], pose[1]),
            s_hint=hint,
            back=cfg.projection_back,
            ahead=cfg.projection_ahead,
        )
        self._last_s = proj.s

        remaining = trajectory.remaining(proj.s)
        distance_to_end = trajectory.distance_to_end((pose[0], pose[1]))
        finished = (
            remaining <= cfg.goal_tolerance
            and distance_to_end <= cfg.goal_tolerance
        )

        lookahead = self.lookahead_distance()
        target = trajectory.lookahead_point(proj.s, lookahead)

        local = to_local_frame(target, pose)
        d_sq = float(local @ local)
        if d_sq < 1e-9:
            curvature = 0.0
        else:
            curvature = 2.0 * float(local[1]) / d_sq

        # Profil de vitesse : croisière, ralentissement en virage, arrivée.
        v_cmd = cfg.max_linear_speed / (1.0 + cfg.curvature_slowdown * abs(curvature))
        if remaining < cfg.slowdown_radius:
            ratio = remaining / cfg.slowdown_radius
            v_cmd = min(v_cmd, max(cfg.min_linear_speed, cfg.max_linear_speed * ratio))
        v_cmd = max(v_cmd, 0.0)

        omega_cmd = v_cmd * curvature
        if abs(omega_cmd) > cfg.max_angular_speed:
            omega_cmd = math.copysign(cfg.max_angular_speed, omega_cmd)
            if abs(curvature) > 1e-9:
                v_cmd = min(v_cmd, cfg.max_angular_speed / abs(curvature))

        if finished:
            v_cmd, omega_cmd = 0.0, 0.0

        self._last_v = v_cmd

        tangent = trajectory.heading_at(proj.s)
        offset = np.array([pose[0], pose[1]]) - proj.point
        cross_track = float(
            math.cos(tangent) * offset[1] - math.sin(tangent) * offset[0]
        )

        return ControlCommand(
            v=v_cmd,
            omega=omega_cmd,
            s=proj.s,
            lookahead_point=target,
            lookahead=lookahead,
            curvature=curvature,
            cross_track_error=cross_track,
            heading_error=wrap_to_pi(pose[2] - tangent),
            remaining=remaining,
            finished=finished,
        )
