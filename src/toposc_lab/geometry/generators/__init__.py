"""Generators for model-independent discrete geometries."""

from toposc_lab.geometry.generators.chain import chain
from toposc_lab.geometry.generators.irregular import irregular_cluster
from toposc_lab.geometry.generators.ring import ring
from toposc_lab.geometry.generators.square import square

__all__ = ["chain", "irregular_cluster", "ring", "square"]
