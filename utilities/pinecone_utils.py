"""Implements an async interface for Pinecone queries."""

from typing import Any, Optional

import httpx
import json

_PINECONE_CONNECT_TIMEOUT = 2

def _get_pinecone_timeout(timeout: int) -> httpx.Timeout:
  """Returns a timeout for Pinecone queries, with a fixed connect timeout."""
  return httpx.Timeout(
    timeout,
    connect=_PINECONE_CONNECT_TIMEOUT,
    read=timeout,
    write=timeout)

_SERVERLESS_ENVIRONMENT_NAME = "gcp-us-central1-4a9f"

def get_serverless_environment(index_name: str,
                               environment: str) -> str:
  if index_name in (
    "usa-guidelines", "european-guidelines", "multilingual-guidelines"):
    return _SERVERLESS_ENVIRONMENT_NAME
  return environment

async def query_pinecone(
    vector: list[float],
    index_name: str,
    api_key: str,
    environment: str,
    project_id: str,
    top_k: int,
    timeout: int = 5,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    pubmed_types: Optional[list[str]] = None,
) -> dict[str, Any]:
  """
  Queries Pinecone for the top k results for a given vector.
  Applies potential filters. Records in the Pinecone db without the "year" 
  metadata field should not be returned if either "year_start" or "year_end" 
  is provided. All records in the "pubmed-abstracts" index should have a 
  "pubmed_types" metadata field that is an array of strings. It should not be 
  empty, default to `["Unknown"]`, so it gets returned when "Other" is selected 
  on the frontend.
  """
  environment = get_serverless_environment(index_name, environment)
  filters = {"$and": []}
  if year_start is not None:
    filters["$and"].append({"year": {"$gte": year_start}})
  if year_end is not None:
    filters["$and"].append({"year": {"$lte": year_end}})
  if index_name == "pubmed-abstracts" \
    and pubmed_types is not None \
    and len(pubmed_types) != 80:
    filters["$and"].append({"pubmed_types": {"$in": pubmed_types}})

  query_payload = {
      "vector": vector,
      "topK": top_k,
  }
  if len(filters["$and"]) > 0:
    query_payload["filter"] = filters

  endpoint = (
      f"https://{index_name}-{project_id}.svc.{environment}.pinecone.io/query")

  async with httpx.AsyncClient() as client:
    response = await client.post(endpoint,
                                 headers={
                                     "Api-Key": api_key,
                                     "Content-Type": "application/json",
                                     "X-Pinecone-API-Version": "2024-07"
                                 },
                                 data=json.dumps(query_payload),
                                 timeout=_get_pinecone_timeout(timeout))

  response.raise_for_status()
  return response.json()
