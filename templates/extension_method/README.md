# Extension Method Template

This template is a non-runtime scaffold for new WAM method authors. Copy the
shape, not the names.

Suggested workflow:

1. Add an enum/config dataclass.
2. Implement a `PolicyVariant`.
3. Implement or reuse an `ActionDecoder`.
4. Register builders through the generic registries.
5. Add one tiny smoke config and focused tests.

Do not import this template from production code.
