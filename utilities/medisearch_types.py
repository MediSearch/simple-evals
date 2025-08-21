

from dataclasses import dataclass

@dataclass
class Article:
  """Represents an article."""
  article_id: str
  title: str
  url: str
  authors: list[str]
  year: str
  abstract: str = ""
  journal: str = ""
  n_citations: int = 0
