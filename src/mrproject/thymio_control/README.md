# thymio_control — paquet prêt à copier

Suivi de trajectoire (sortie d'A*) pour le Thymio, à partir de la seule pose
`(x, y, theta)` fournie par la perception.

## Installation

Copier le dossier `thymio_control/` dans le `src/` du projet :

```
projet/
└── src/
    └── thymio_control/
        ├── __init__.py
        ├── controller.py
        ├── geometry.py
        ├── kinematics.py
        └── trajectory.py
```

C'est tout : aucun `pip install`, aucun fichier de build. Le paquet est
importable dès que `src/` est sur le `PYTHONPATH` (c'est le cas par défaut avec
un layout `src/`).

## Dépendance

`numpy` uniquement. Ni matplotlib, ni pytest, ni `tdmclient` : le paquet est
embarquable tel quel sur le robot. L'envoi des consignes moteur reste à la
charge de l'appelant.

```
pip install numpy
```

## Usage minimal

```python
from thymio_control import Trajectory, PurePursuitController

path = Trajectory([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
controller = PurePursuitController()
controller.reset()

command = controller.compute((0.0, 0.0, 0.0), path)
print(command.v, command.omega, command.finished)
```

## Contrat d'appel

- `compute(pose, trajectory)` prend la pose **à chaque appel** : le contrôleur
  ne la mémorise pas. Une nouvelle estimation de pose est donc acceptée
  directement, sans rien réinitialiser.
- Un nouveau chemin doit être un **nouvel objet** `Trajectory` (les longueurs
  cumulées sont calculées dans `__init__`). Ne jamais muter `.points` : la
  propriété renvoie le tableau interne sans copie, ce qui désynchronise la
  géométrie en silence.
- Après un changement de chemin, appeler `controller.reset()` : `_last_s`
  mémorise l'abscisse curviligne de l'ancien chemin et sert de `s_hint` pour
  restreindre la projection à `[s - 0.30, s + 0.60]`. Sans reset, la projection
  peut sauter sur une autre portion du nouveau chemin.
- Aucun horodatage n'est porté par `Pose` ni `Trajectory` : la détection d'une
  pose périmée et l'arbitrage « dernière valeur gagnante » sont à la charge de
  la boucle appelante.

## Contenu

| Module | Rôle |
|---|---|
| `geometry.py` | `Pose`, changements de repère, `wrap_to_pi`, `distance` |
| `trajectory.py` | `Trajectory` (polyligne + abscisse curviligne), `Projection` |
| `controller.py` | `PurePursuitController`, `PurePursuitConfig`, `ControlCommand` |
| `kinematics.py` | `DifferentialDrive`, `integrate_unicycle` |
