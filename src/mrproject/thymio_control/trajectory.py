"""Trajectoire pré-calculée (sortie d'A*) et projection d'une pose dessus.

Une trajectoire est une polyligne ouverte décrite par une séquence de points
``(x, y)`` dans le repère monde. On lui associe une abscisse curviligne ``s``
qui croît de ``0`` (départ) à ``length`` (arrivée), ce qui permet de raisonner
en « distance parcourue le long du chemin » plutôt qu'en index de point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import Pose, Vec2, distance, wrap_to_pi

Vec2Array = NDArray[np.float64]


@dataclass(frozen=True)
class Projection:
    """Résultat de la projection orthogonale d'un point sur la trajectoire."""

    s: float
    """Abscisse curviligne du point projeté, dans ``[0, length]``."""

    point: Vec2
    """Point de la trajectoire le plus proche."""

    distance: float
    """Distance (erreur latérale) entre le point projeté et le point d'origine."""

    segment: int
    """Index du segment sur lequel tombe la projection."""


class Trajectory:
    """Polyligne ouverte avec abscisse curviligne.

    Parameters
    ----------
    points:
        Séquence de points ``(x, y)``, au moins deux points distincts.
    dedupe:
        Supprime les points consécutifs identiques (fréquents dans les sorties
        d'A* ou après un ré-échantillonnage).
    """

    def __init__(self, points: ArrayLike, *, dedupe: bool = True) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                f"points doit être de forme (N, 2), reçu {pts.shape}"
            )
        if not np.all(np.isfinite(pts)):
            raise ValueError("points contient des valeurs non finies")

        if dedupe and len(pts) > 1:
            keep = np.concatenate(
                [[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]
            )
            pts = pts[keep]

        if len(pts) < 2:
            raise ValueError("une trajectoire nécessite au moins deux points distincts")

        self._points = pts
        self._segments = np.diff(pts, axis=0)
        self._segment_lengths = np.linalg.norm(self._segments, axis=1)
        self._segment_lengths_sq = self._segment_lengths**2
        self._cumulative = np.concatenate(
            [[0.0], np.cumsum(self._segment_lengths)]
        )

    # ------------------------------------------------------------------
    # Propriétés
    # ------------------------------------------------------------------
    @property
    def points(self) -> Vec2Array:
        """Points de la polyligne, forme ``(N, 2)``."""
        return self._points

    @property
    def cumulative(self) -> Vec2Array:
        """Abscisse curviligne cumulée à chaque sommet, forme ``(N,)``."""
        return self._cumulative

    @property
    def length(self) -> float:
        """Longueur totale du chemin, en mètres."""
        return float(self._cumulative[-1])

    @property
    def start(self) -> Vec2:
        """Point de départ."""
        return self._points[0].copy()

    @property
    def end(self) -> Vec2:
        """Point d'arrivée."""
        return self._points[-1].copy()

    def __len__(self) -> int:
        return len(self._points)

    def __repr__(self) -> str:
        return (
            f"Trajectory(n_points={len(self._points)}, length={self.length:.3f} m)"
        )

    # ------------------------------------------------------------------
    # Échantillonnage
    # ------------------------------------------------------------------
    def point_at(self, s: float) -> Vec2:
        """Point situé à l'abscisse curviligne ``s`` (bornée aux extrémités)."""
        s = float(np.clip(s, 0.0, self.length))
        idx = int(np.searchsorted(self._cumulative, s, side="right") - 1)
        idx = min(max(idx, 0), len(self._segments) - 1)
        local = s - self._cumulative[idx]
        t = local / self._segment_lengths[idx]
        return self._points[idx] + t * self._segments[idx]

    def heading_at(self, s: float) -> float:
        """Cap (tangente) de la trajectoire à l'abscisse ``s``."""
        s = float(np.clip(s, 0.0, self.length))
        idx = int(np.searchsorted(self._cumulative, s, side="right") - 1)
        idx = min(max(idx, 0), len(self._segments) - 1)
        seg = self._segments[idx]
        return float(np.arctan2(seg[1], seg[0]))

    def lookahead_point(self, s: float, lookahead: float) -> Vec2:
        """Point situé ``lookahead`` mètres plus loin que ``s``."""
        return self.point_at(s + max(lookahead, 0.0))

    def resample(self, step: float) -> "Trajectory":
        """Ré-échantillonne la trajectoire à pas d'arc constant."""
        if step <= 0:
            raise ValueError("step doit être strictement positif")
        count = max(int(np.ceil(self.length / step)), 1)
        s_values = np.linspace(0.0, self.length, count + 1)
        return Trajectory(np.array([self.point_at(s) for s in s_values]))

    def smoothed(self, window: int = 5) -> "Trajectory":
        """Lisse la polyligne par moyenne glissante (extrémités conservées).

        Utile pour adoucir les angles droits d'une sortie A* sur grille, au prix
        d'un léger rabotage des virages : à n'utiliser que si la marge vis-à-vis
        des obstacles le permet.
        """
        if window < 3:
            return Trajectory(self._points.copy())
        kernel = np.ones(window, dtype=float) / window
        padded = np.pad(
            self._points, ((window // 2, window // 2), (0, 0)), mode="edge"
        )
        smoothed = np.column_stack(
            [
                np.convolve(padded[:, 0], kernel, mode="valid"),
                np.convolve(padded[:, 1], kernel, mode="valid"),
            ]
        )
        smoothed[0] = self._points[0]
        smoothed[-1] = self._points[-1]
        return Trajectory(smoothed)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def project(
        self,
        point: ArrayLike,
        *,
        s_hint: float | None = None,
        back: float = 0.30,
        ahead: float = 0.60,
    ) -> Projection:
        """Projette un point sur la trajectoire.

        Parameters
        ----------
        point:
            Position à projeter, typiquement ``(x, y)`` du robot.
        s_hint:
            Dernière abscisse connue. Si fournie, la recherche est restreinte à
            ``[s_hint - back, s_hint + ahead]`` : cela évite de « sauter » sur
            une portion voisine du chemin (virages serrés, chemin qui repasse
            près de lui-même) et coûte beaucoup moins cher.
        back, ahead:
            Fenêtre de recherche autour de ``s_hint``, en mètres.
        """
        p = np.asarray(point, dtype=float).reshape(2)

        if s_hint is None:
            lo, hi = 0, len(self._segments) - 1
        else:
            s_lo = max(0.0, float(s_hint) - back)
            s_hi = min(self.length, float(s_hint) + ahead)
            lo = int(np.searchsorted(self._cumulative, s_lo, side="right") - 1)
            hi = int(np.searchsorted(self._cumulative, s_hi, side="left"))
            lo = min(max(lo, 0), len(self._segments) - 1)
            hi = min(max(hi, lo), len(self._segments) - 1)

        starts = self._points[lo : hi + 1]
        segs = self._segments[lo : hi + 1]
        denom = self._segment_lengths_sq[lo : hi + 1]

        t = np.clip(((p - starts) * segs).sum(axis=1) / denom, 0.0, 1.0)
        closest = starts + t[:, None] * segs
        dists = np.linalg.norm(p - closest, axis=1)

        best = int(np.argmin(dists))
        idx = lo + best
        s = float(self._cumulative[idx] + t[best] * self._segment_lengths[idx])
        return Projection(
            s=s,
            point=closest[best].copy(),
            distance=float(dists[best]),
            segment=idx,
        )

    def cross_track_error(self, pose: Pose, *, s_hint: float | None = None) -> float:
        """Erreur latérale signée : positive si le robot est à gauche du chemin."""
        proj = self.project((pose[0], pose[1]), s_hint=s_hint)
        tangent = self.heading_at(proj.s)
        offset = np.array([pose[0], pose[1]]) - proj.point
        # Produit vectoriel tangente x offset : > 0 => offset à gauche.
        return float(
            np.cos(tangent) * offset[1] - np.sin(tangent) * offset[0]
        )

    def heading_error(self, pose: Pose, *, s_hint: float | None = None) -> float:
        """Écart angulaire entre le cap du robot et la tangente du chemin."""
        proj = self.project((pose[0], pose[1]), s_hint=s_hint)
        return wrap_to_pi(pose[2] - self.heading_at(proj.s))

    def remaining(self, s: float) -> float:
        """Distance restante le long du chemin depuis l'abscisse ``s``."""
        return max(self.length - float(s), 0.0)

    def distance_to_end(self, point: ArrayLike) -> float:
        """Distance euclidienne directe au point d'arrivée."""
        return distance(point, self.end)
