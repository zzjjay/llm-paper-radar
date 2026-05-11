from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.semantic_scholar import SemanticScholarSource

FIXTURE = Path(__file__).parent / "fixtures" / "ss_citations_response.json"


@respx.mock
@pytest.mark.asyncio
async def test_ss_fetches_citations_for_seeds(tmp_path: Path):
    seeds_file = tmp_path / "seeds.yaml"
    seeds_file.write_text("seeds:\n  - id: arXiv:2210.17323\n    name: GPTQ\n")

    respx.get("https://api.semanticscholar.org/graph/v1/paper/arXiv:2210.17323/citations").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )

    src = SemanticScholarSource(seeds_file=seeds_file, citation_window_days=7)
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))

    assert len(papers) == 1
    p = papers[0]
    assert p.id == "2405.12345"
    assert p.sources[0].name == "semantic_scholar"
