import pytest

try:
    import PyQt6.QtWebEngineWidgets  # noqa: F401
except ImportError:
    pytest.skip(
        "timeline UI tests require a loadable Qt WebEngine runtime",
        allow_module_level=True,
    )

from coopinfer.app import MainWindow, _timeline_short_node_label


def test_timeline_labels_are_compact():
    assert _timeline_short_node_label("layer_s02_l010::step000") == "S2/L10/D0"
    assert _timeline_short_node_label("layer_s02_l010::step009[f3]") == "S2/L10/D9/F3"


def test_timeline_page_defaults_to_compact_labels_and_keeps_hover_mode():
    page = MainWindow._svg_page(
        None,
        "Timeline",
        '<g><title>full details</title><rect width="8"/><text class="timeline-label">S1/L2</text></g>',
        640,
        240,
        scale_axis="timeline",
        timeline_left_pad=120.0,
        timeline_px_per_ms=9.0,
        timeline_right_pad=80.0,
        timeline_total_ms=10.0,
    )

    assert '<option value="auto" selected>精简</option>' in page
    assert '<option value="none">仅悬停</option>' in page
    assert "barWidth >= textWidth" in page
    assert "updateTimelineLabels();" in page
    assert '<button id="zoomFit" type="button">适应窗口</button>' in page
    assert 'id="pxPerMs"' in page
    assert "viewport.addEventListener('wheel'" in page
    assert "event.ctrlKey" in page


def test_non_timeline_page_has_no_timeline_label_controls():
    page = MainWindow._svg_page(None, "Graph", "<g />", 640, 240, scale_axis="x")
    assert 'id="labelMode"' not in page
    assert 'id="zoomFit"' not in page
    assert 'id="pxPerMs"' not in page
