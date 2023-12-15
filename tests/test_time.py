import irfuplanets.time


def test_import():
    assert isinstance(irfuplanets.time.now(), float)
