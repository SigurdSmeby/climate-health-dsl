"""Phase 0 smoke test: the installed package can be imported at all."""


def test_import():
    import dsl  # noqa: F401  (importing is the whole test)
