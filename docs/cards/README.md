# Artifact And Experiment Cards

Cards are the public reproducibility layer. Each public checkpoint, dataset
fixture, simulator contract, or benchmark recipe should have a short card that
states what it is, how to validate it, what resources it needs, and what it
does not claim.

Required fields:

- method family or fixture family
- variant
- benchmark or dataset
- config path
- artifact manifest id, when applicable
- action/state/visual dimensions
- resource requirements
- validation command
- expected metrics or contract outcome
- license/source
- known limitations

Cards that point at private or unpublished checkpoints must say so explicitly.
