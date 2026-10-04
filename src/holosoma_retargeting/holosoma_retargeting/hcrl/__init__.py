"""Adapters from external motion corpora (and fitted arena courts) to holosoma retargeting inputs.

The pseudo-source route: BONES-SEED csvs are already-IK'd G1 motions, so the "human" source fed to the
interaction-mesh retargeter is the FK of those csvs (the ``g1fk`` data format, identity joint mapping,
scale 1). The constrained solve then re-projects each clip onto its reconstructed court with contact,
non-penetration and velocity constraints.
"""
