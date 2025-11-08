from pathlib import Path

from sematic_desktop.routing import ConversionRouter, gather_file_signals


def test_router_prefers_docling_for_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "slides.pdf"
    pdf.write_bytes(b"binary" * 100)

    router = ConversionRouter()
    signals = gather_file_signals(pdf)
    order = router.plan_order(signals)

    assert order[0] == "docling"


def test_router_quality_scoring_increases_with_richer_markdown(tmp_path: Path) -> None:
    doc = tmp_path / "large.docx"
    doc.write_bytes(b"x" * 10_000)

    router = ConversionRouter()
    signals = gather_file_signals(doc)

    poor = router.score_markdown("brief text", signals)
    rich = router.score_markdown("## Heading\n" + ("content row | value\n" * 200), signals)

    assert poor < rich
    assert router.is_quality_acceptable(rich, signals)
