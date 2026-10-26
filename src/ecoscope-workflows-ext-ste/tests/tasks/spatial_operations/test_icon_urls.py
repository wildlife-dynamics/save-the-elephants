from unittest import mock

import pytest

from ecoscope_workflows_ext_ste.tasks.results._spatial_layers import _UnifiedLegend
from ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_features import _server_icon_url

SERVER = "https://lg.pamdas.org"


@pytest.mark.parametrize(
    "image, expected",
    [
        ("/static/sprite-src/lion.svg", "https://lg.pamdas.org/static/sprite-src/lion.svg"),
        ("static/lion.svg", "https://lg.pamdas.org/static/lion.svg"),
        ("https://lg.pamdas.org/static/lion.svg", "https://lg.pamdas.org/static/lion.svg"),
        ("https://evil.example/x.svg", None),  # another host
        ("//evil.example/x.svg", None),  # protocol-relative URL to another host
        ("http://lg.pamdas.org/static/lion.svg", None),  # not https
        ("", None),
        (None, None),
    ],
)
def test_server_icon_url(image, expected):
    assert _server_icon_url(SERVER, image) == expected


def test_server_icon_url_needs_an_https_server():
    assert _server_icon_url("http://lg.pamdas.org", "/static/lion.svg") is None
    assert _server_icon_url("", "/static/lion.svg") is None


def test_fetch_svg_verifies_tls_and_needs_https():
    with mock.patch("ecoscope_workflows_ext_ste.tasks.results._spatial_layers.requests.get") as get:
        get.return_value.text = '<svg fill="#ff0000"/>'
        assert _UnifiedLegend._fetch_svg("https://lg.pamdas.org/static/lion.svg") is not None
        assert "verify" not in get.call_args.kwargs  # requests verifies certificates by default
        assert _UnifiedLegend._fetch_svg("http://lg.pamdas.org/static/lion.svg") is None
        assert get.call_count == 1
