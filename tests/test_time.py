def test_import():
    import irfuplanets.time

    assert isinstance(irfuplanets.time.now(), float)
