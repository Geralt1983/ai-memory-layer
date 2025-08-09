import os
from typing import List
from integrations.embeddings_interfaces import EmbeddingProvider
from integrations.providers.dualwrite import DualWriteEmbeddings

class _P(EmbeddingProvider):  # primary returns simple ascending vectors
    def embed(self, texts: List[str]):
        return [[float(i), 0.0] for i, _ in enumerate(texts)]

class _S(EmbeddingProvider):  # shadow returns slightly different vectors
    def embed(self, texts: List[str]):
        return [[float(i)+0.1, 0.0] for i, _ in enumerate(texts)]

def test_ab_metrics_csv_written(tmp_path, monkeypatch):
    csv_path = tmp_path/"ab.csv"
    os.environ["EMBED_AB_LOG"] = str(csv_path)
    try:
        e = DualWriteEmbeddings(primary=_P(), shadow=_S())
        out = e.embed(["a","b","c"])
        # primary result returned
        assert out == [[0.0,0.0],[1.0,0.0],[2.0,0.0]]
        # file written
        assert csv_path.exists()
        content = csv_path.read_text().strip().splitlines()
        # headers + one row
        assert len(content) == 2
        # header sanity
        assert "mean_cosine" in content[0]
    finally:
        os.environ.pop("EMBED_AB_LOG", None)