from unittest.mock import MagicMock

from cli_interface import MemoryLayerCLI


def test_delete_memory_success():
    cli = MemoryLayerCLI("http://test")
    mock_resp = MagicMock(status_code=204, text="")
    cli.session.delete = MagicMock(return_value=mock_resp)

    assert cli.delete_memory("123") is True
    cli.session.delete.assert_called_once_with("http://test/memories/123")


def test_update_memory_success():
    cli = MemoryLayerCLI("http://test")
    mock_resp = MagicMock(status_code=201, text="")
    cli.session.put = MagicMock(return_value=mock_resp)

    content = "updated"
    metadata = {"a": 1}

    assert cli.update_memory("abc", content, metadata) is True
    cli.session.put.assert_called_once_with(
        "http://test/memories/abc", json={"content": content, "metadata": metadata}
    )

