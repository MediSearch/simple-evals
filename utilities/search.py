from .medisearch_types import Article
from .scrape import scrape
import httpx

import json

_RERANK_ENDPOINT = ("https://Cohere-rerank-v3-5-cdivc.eastus2.models.ai"
                    ".azure.com/v1/rerank")
_RERANK_TIMEOUT = 120

def articles_to_json(articles: list[Article]) -> str:
  data = []
  for i, article in enumerate(articles, start=1):
    data.append({
      "id": i,
      "title": article.title,
      "snippet": article.abstract,   # previously "Snippet"
      "url": article.url,
      "year": article.year,
      "journal": article.journal,
      "n_citations": article.n_citations
    })
  return json.dumps(data)

async def get_cohere_rerank(
    query: str,
    documents: list[str],
    top_n: int,
) -> tuple[list[int], list[float]]:
  try:
    data = {
      "documents": documents,
      "query": query,
      "top_n": top_n
    }
    body = str.encode(json.dumps(data))
    headers = {
      "Content-Type": "application/json",
      "Accept": "application/json",
      "Authorization": ("Bearer " + "7u9Ph0MZfLKcMxOZeqH6jLSK4j7m1smP"),
    }

    async with httpx.AsyncClient() as client:
      response = await client.post(
        _RERANK_ENDPOINT,
        headers=headers,
        data=body,
        timeout=_RERANK_TIMEOUT
      )
      response.raise_for_status()

      results = response.json()
      indices = [result["index"] for result in results["results"]]
      scores = [result["relevance_score"] for result in results["results"]]
      return indices, scores

  except httpx.ReadTimeout as exception:
    raise exception


async def rank_articles_with_reranker(
  query: str,
  articles: list[Article],
) -> list[Article]:
  """Rank articles based on a query.

    Args:
        query: Query to use for ranking.
        articles: List of articles to rank.
    Returns:
        list[Article] of ranked articles.
    """

  snippets = [f"Title: {article.title}\nSnippet: {article.abstract}\nJournal: {article.journal}\nYear: {article.year}" for article in articles]
  for i, article in enumerate(articles):
    article.snippet = snippets[i]
  indices, scores = await get_cohere_rerank(query=query,
    documents=snippets, top_n=len(snippets))
  results = []
  for index, _ in zip(indices, scores):
    article = articles[index]
    results.append(article)

  return results

async def search_articles(query: str,
                          n_results: int = 20) -> str:
  articles = await scrape(query)
  articles = await rank_articles_with_reranker(query, articles)
  return articles_to_json(articles[:n_results])
