"""Smoke tests for the endpoint wrapper. No live model calls."""

from gvc_local.endpoint import EndpointConfig, LLAMA_31_8B, QWEN_25_7B


def test_llama_constructor():
    cfg = EndpointConfig.llama31_8b(base_url="http://foo:8000/v1", default_temperature=0.5)
    assert cfg.model == LLAMA_31_8B
    assert cfg.base_url == "http://foo:8000/v1"
    assert cfg.default_temperature == 0.5


def test_qwen_constructor():
    cfg = EndpointConfig.qwen25_7b()
    assert cfg.model == QWEN_25_7B
    assert cfg.api_key == "EMPTY"


def test_client_factory_does_not_call_api():
    cfg = EndpointConfig.llama31_8b()
    client = cfg.client()
    assert client.cfg is cfg
