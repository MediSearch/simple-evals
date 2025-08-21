"""Fetches articles across all MediSearch indexes."""

from typing import Optional
import asyncio


import cohere
import httpx
from google.cloud import spanner
from google.api_core.exceptions import DeadlineExceeded

from .medisearch_types import Article
from .pinecone_utils import query_pinecone

# Initialize Spanner client
_COHERE_CLIENT = cohere.Client('8pGg3e7YT5TK5TrOQSLjNyQQtQDVywFRuPHQIvWK',
                                    timeout=5)
_DATABASE = spanner.Client().instance(
  "pubmed-abstracts").database("pubmed-abstracts")


def _fetch_articles_from_spanner(
        article_ids: list[str],
        timeout: int) -> list[Optional[tuple[str, ...]]]:
  """Fetches articles from Spanner database based on a list of article IDs."""

  # SQL query to fetch articles by ID
  query ="""
  SELECT id, title, abstract, authors, year, url, type, 
  journal, n_citations, pubmed_type, tldr
  FROM articles
  WHERE id IN UNNEST(@article_ids)
  """
  parameters = {"article_ids": article_ids}
  param_types = {"article_ids":
                 spanner.param_types.Array(spanner.param_types.STRING)}

  # Execute the query and fetch results
  try:
    with _DATABASE.snapshot(multi_use=True) as snapshot:
      fetch_results = snapshot.execute_sql(
          query, params=parameters, param_types=param_types, timeout=timeout)
      articles = []
      for row in fetch_results:
        articles.append(row)
  except DeadlineExceeded as e:
    print("Got into error", e)

  # Fill in None for articles not found
  return [next((article for article in articles if article[0] == article_id),
               None) for article_id in article_ids]


async def get_article(uuid: str) -> Optional[dict[str, str]]:
  """Fetches an article from Spanner database based on a UUID."""

  try:
    articles = _fetch_articles_from_spanner([uuid], 5)
  except RuntimeError as _:
    return None

  if not articles or not articles[0]:
    return None

  article = articles[0]
  return {
    "title": article[1],
    "abstract": article[2],
    "authors": article[3],
    "year": article[4],
    "url": article[5],
    "journal": article[7],
    "tldr": article[10],
  }


async def _fetch_indices(
    query_vector: list[float],
    sources: tuple[str, ...],
    source_count_limits: dict[str, int],
    timeout: int) -> dict[str, float]:
  """Parse Pinecone indices for relevant articles."""

  try:
    tasks = [query_pinecone(
              query_vector,
              source,
              '30f34db1-2757-4e02-87f2-89c18c26007e',
              'us-central1-gcp',
              'zubx6am',
              source_count_limits[source],
              timeout
          ) for source in sources]

    pinecone_results = await asyncio.gather(*tasks)
  except httpx.ReadTimeout as exception:
    raise exception

  return {fetch_result["id"]: fetch_result["score"]
                for result in pinecone_results
                for fetch_result in result["matches"]}


def _get_query_embedding(query: str) -> list[float]:
  try:
    response = _COHERE_CLIENT.embed(
      [query], model="embed-english-v3.0", input_type="search_query")
    query_embedding = response.embeddings[0]
  except cohere.CohereError as exception:
    raise exception

  return query_embedding


async def scrape(query: str, # pylint: disable=dangerous-default-value
                 sources: tuple[str, ...] = (
                  "pubmed-abstracts",
                  "drugs",
                  "healthline",
                  "medline",
                  "books",
                  "guidelines",
                  "authoritative-guidelines",
                  "usa-guidelines",
                  "european-guidelines",
                  ),
                 source_count_limits: dict[str, int] = {
                  "pubmed-abstracts": 120,
                  "books": 30,
                  "guidelines": 30,
                  "authoritative-guidelines": 30,
                  "usa-guidelines": 30,
                  "european-guidelines": 30,
                  "drugs": 5,
                  "healthline": 5,
                  "medline": 5,
                 },
                 timeout: int = 5) -> list[Article]:
  """Asynchronously scrapes drugs articles."""

  query_embedding = _get_query_embedding(query)
  ids_to_scores = await _fetch_indices(
    query_embedding,
    sources,
    source_count_limits,
    timeout
  )

  ids = list(ids_to_scores.keys())
  spanner_articles = _fetch_articles_from_spanner(ids, timeout)

  url_abstract_set = set()
  scraping_results = []
  for article in spanner_articles:
    if article is None:
      continue

    if not article[6]:
      continue

    title = article[1]
    if not title or title == "":
      continue

    scraped_article = Article(
          title=title,
          article_id=article[0],
          url=article[5],
          abstract=article[2],
          authors=article[3],
          year=article[4],
          journal=article[7],
          n_citations=article[8],
    )

    if (scraped_article.url) and (scraped_article.url != "") \
      and (scraped_article.url, scraped_article.abstract) in url_abstract_set:
      continue

    url_abstract_set.add((scraped_article.url, scraped_article.abstract))
    scraping_results.append(scraped_article)

  return sorted(
    scraping_results, key=lambda x: ids_to_scores[x.article_id], reverse=True)
