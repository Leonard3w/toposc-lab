# Phase 6.7 decision: explicit geometry dimension records

Date: 2026-08-30  
Status: accepted

## Decision

Geometry dimensions other than coordinate embedding are represented by immutable
`GeometryDimension` records. Every record contains:

- an explicit kind;
- a numeric value;
- whether it applies to the finite geometry or its infinite family;
- the method or definition that produced the value;
- whether the value is exact.

`Geometry.embedding_dimension` remains the sole description of coordinate-space
dimension. Dimension records are never interpreted implicitly as the physical dimension
for topology dispatch.

## Migration

The ambiguous free-form `intrinsic_dimension` metadata key is removed. Regular chain,
ring, square, triangular, honeycomb, Kagome, cubic, and BCC generators now carry an exact
`lattice` dimension obtained from translation rank. The irregular reference cluster has
no scalar dimension record because its two coordinate columns do not establish an
intrinsic or effective graph dimension.

The Sierpiński gasket carries two separate exact infinite-family records:

- topological covering dimension `1`;
- Hausdorff dimension `log(3) / log(2)` from analytic self-similarity.

Finite-size estimates such as box-counting, spectral, or walk dimensions must use their
own kinds and methods in later phases rather than overwriting either value.

## Consequences

- Embedded dimension, lattice rank, topology dimension, and fractal dimension cannot be
  confused through one metadata field.
- Automated discovery can retain multiple dimension estimates with explicit provenance.
- A model or experiment must still declare the physical dimension used by topology
  dispatch.
- Geometry serialization must persist dimension records explicitly in Phase 6.23.
