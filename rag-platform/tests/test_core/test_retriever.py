import core


def test_core_exports_retriever_loader():
    assert callable(core.load_retriever)
