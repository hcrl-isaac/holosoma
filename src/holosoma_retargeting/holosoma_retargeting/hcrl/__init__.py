"""hcrl adapter: BONES-SEED G1 csvs + fitted arena courts -> holosoma climbing-task inputs.

The pseudo-source route: our csvs are already-IK'd G1 motions, so the "human" source fed to the
interaction-mesh retargeter is the FK of those csvs (the ``g1fk`` data format, identity joint mapping,
scale 1). The constrained solve then re-projects each clip onto its reconstructed court with hard
contact / non-penetration / velocity constraints -- replacing the greedy snap+IK cleanup pipeline.
"""
