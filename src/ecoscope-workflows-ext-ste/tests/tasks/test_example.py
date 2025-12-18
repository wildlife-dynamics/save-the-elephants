from ecoscope_workflows_ext_ste.tasks import add_two_thousand


def test_add_two_thousand():
    assert 2001 == add_two_thousand(1)
