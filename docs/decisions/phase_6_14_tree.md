# Phase 6.14 decision: general canonical rooted tree

## Scope and separation from Phase 6.15

Phase 6.14 represents an arbitrary finite rooted tree supplied by its parent
sequence. It does not assume constant branching or coordination. The symmetric
Cayley/Bethe-like construction remains a separate generator in Phase 6.15 and
will reuse the same rooted hierarchy representation.

## Typed hierarchy

`RootedTreeStructure` stores one parent per site, the root, canonical child
lists, depths, and leaves. It rejects invalid indices, multiple roots, cycles,
self-parenting, and branches that do not descend from the root. When attached
to `Geometry`, its parent edges must exactly equal the graph edge set. Rooted
semantics therefore cannot drift into ambiguous free-form metadata.

## Generator convention

`tree(parents)` fixes site 0 as root. `parents[i - 1]` is the parent of site
`i` and must satisfy `0 <= parent < i`. This parent-before-child convention
provides deterministic site order and parent-to-child edge orientation while
remaining able to represent every finite rooted tree topology after canonical
relabeling.

Edges are typed `tree_branch`. Sites are typed `root`, `internal`, or `leaf`;
the root retains its root type even for a singleton. Sites without children are
the boundary sites, including the singleton root. No spatial coordinates or
dimension records are inferred for the abstract tree.
