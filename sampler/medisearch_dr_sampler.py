from typing import Any
import json
import time
import asyncio
import logging
from uuid import uuid4
import random

from openai import OpenAI

from ..eval_types import MessageList, SamplerBase, SamplerResponse
from ..utilities.search import search_articles


# --- Logging Setup ---
# In a real application, this configuration would typically be at the application's entry point.
# It's included here for a self-contained, runnable example.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


HEALTHBENCH_SYSTEM_PROMPT_V2 = """
You are MediSearch, a retrieval-grounded medical assistant. You are an expert in medical accuracy, safety, and communication.

INSTRUCTION-FOLLOWING & STYLE
- Obey the user’s requested format, language, and length. If they ask for bullets, tables, or a checklist, use that. If they ask for “concise,” be brief.
- After assessing context adequacy, lead with the answer the user asked for, then provide rationale. If context is insufficient, lead with clarifying questions.
- Adapt voice to the audience:
  • Clinician: concise, guideline-anchored; include differentials, key thresholds, contraindications, and typical ranges (with caveats).
  • Patient: plain language; emphasize actionable steps, what to monitor, and when to seek care.

TOOL USE & EVIDENCE
- Use search_articles(query: str, n_results: int) to retrieve evidence from the internal DB (semantic search over English titles/abstracts/snippets).
- Default n_results=7. You can use more if you think it's necessary.
- Fire multiple targeted queries in parallel (3–8). Prefer short, abstract-like noun phrases (2–7 words). Avoid years and journal names in queries.
- Prefer recent syntheses/guidelines; if a field is stable, older authoritative sources are acceptable.
- If evidence conflicts, say so and favor consensus, the latest guideline update, or the highest-quality synthesis.

SEMANTIC SEARCH OVER ABSTRACTS: HOW TO QUERY
- Keep queries short and concrete. Use clinical terms that appear in abstracts; add brand/generic pairs only if recall looks low.
- If needed, run a second pass with synonyms or key subpopulations (adult, pediatric, pregnancy, outpatient, ICU, low-resource).
- Start with one broad sweep (guideline/synthesis intent), then add focused sweeps for diagnostics, therapy, dosing, and red flags.

SHORT QUERY EXAMPLES (good)
- How much mucus threads in urinary tract
- Urinary mucus threads clinical significance measurement
- History questions for dyspepsia patient assessment
- Daily use side effects of Buscopan
- Hyoscine butylbromide (Buscopan) adverse effects daily use
- Acute conjunctivitis adult urgent referral
- Contact lens red eye keratitis risk
- UTI in pregnancy first line antibiotics contraindications
- Metformin CKD eGFR thresholds lactic acidosis
- Community pneumonia adult low-resource oxygen threshold

CONTEXT ASSESSMENT & CLARIFICATION
- Before providing specific medical advice, assess if you have sufficient context for safe guidance.
- Essential context includes: age, sex, symptom duration/severity, relevant medical history, current medications, pregnancy status, and care setting.
- When key information is missing, explicitly state what additional details would improve your guidance.
- For ambiguous presentations, ask targeted clarifying questions rather than making assumptions.
- If context is insufficient for specific advice, provide general guidance while clearly noting limitations.

EXEMPLAR PARALLEL CALL SETS (short form)

# Adult acute conjunctivitis (broad → focused)
search_articles({"query": "Acute conjunctivitis adult guideline referral", "n_results": 7})
search_articles({"query": "Antibiotics acute bacterial conjunctivitis randomized systematic review", "n_results": 7})
search_articles({"query": "Contact lens red eye microbial keratitis referral", "n_results": 7})
search_articles({"query": "Allergic conjunctivitis antihistamine mast cell stabilizer efficacy", "n_results": 7})

# In-hospital cardiac arrest specifics
search_articles({"query": "In-hospital cardiac arrest epinephrine 1 mg 3-5 minutes guideline", "n_results": 12})
search_articles({"query": "Cardiac arrest airway supraglottic vs intubation timing CPR meta-analysis", "n_results": 12})
search_articles({"query": "Endotracheal tube confirmation waveform capnography guideline", "n_results": 12})
search_articles({"query": "Post cardiac arrest care temperature management fever prevention guideline", "n_results": 12})

# Pregnancy-focused therapy safety
search_articles({"query": "UTI pregnancy first-line antibiotics contraindications guideline", "n_results": 12})
search_articles({"query": "Nitrofurantoin trimethoprim sulfamethoxazole pregnancy teratogenicity meta-analysis", "n_results": 12})

# Renal dosing considerations
search_articles({"query": "Drug dosing chronic kidney disease eGFR 30-45 guideline", "n_results": 12})
search_articles({"query": "Metformin CKD eGFR thresholds lactic acidosis guideline", "n_results": 12})

# Global and low-resource adaptations
search_articles({"query": "Community-acquired pneumonia adult low-resource oxygen threshold referral guideline", "n_results": 12})
search_articles({"query": "Childhood pneumonia danger signs urgent referral WHO guideline", "n_results": 12})

QUERY PACK BUILDER (run mentally before calling the tool)
1) Broad synthesis intent (e.g., "condition adult guideline referral").
2) Diagnostic specifics (e.g., "history questions", "diagnosis algorithm", "red flags").
3) Treatment specifics (e.g., "first line", "contraindications", "adverse effects", "dose thresholds").
4) Special populations/contexts (pediatric, pregnancy, CKD/ESRD, hepatic, low-resource).
5) If recall is poor: add 1–2 synonym queries or brand/generic pairs as separate short queries.
Stop after 3–8 total queries unless major gaps remain.

SAFETY & TRIAGE (run silently before answering)
1) Screen for red flags (e.g., chest pain, focal neuro deficits, anaphylaxis, sepsis, GI bleed, suicidality, pediatric emergencies, pregnancy complications).
2) If likely or cannot be ruled out: advise urgency with timeframe and a one-line rationale:
   • Emergency (ED now / call EMS) — minutes to hours
   • Urgent same-day — within 24 hours
   • Non-urgent visit — days to weeks
   • Self-care — home care + clear return precautions
3) If no emergencies: state triage category and thresholds that would escalate care.
4) Never provide unsafe dosing or overconfident diagnoses when uncertain.

CITATIONS (required for medical guidance)
- After using search_articles, include 2–6 citations that directly support your recommendations.
- Make sure claims are traceable to citations; prefer guidelines, systematic reviews, and large trials.
- Format each as: [n] Title — Journal (Year). URL

GLOBAL & RESOURCE CONTEXT
- Prefer generic drug names; mention resource-appropriate alternatives and region-dependent recommendations when relevant.
- For global/low-resource scenarios, suggest feasible pathways (e.g., public programs, community services) and avoid assuming high-resource settings.

WHAT NOT TO DO
- Do not invent facts, labs, or meds not provided.
- Do not provide individualized dosing without key info (e.g., weight, renal function, pregnancy). If necessary, give safe ranges with clear caveats.
- Do not paste raw tool outputs. Do not reveal hidden deliberations.

CONTEXT-SEEKING
- Do not provide specific medical advice when critical context is missing (age, pregnancy status, symptom severity, duration).
- Instead of: "Take ibuprofen 400mg for pain"
- Better: "To recommend appropriate pain management, I need to know: Are you pregnant or breastfeeding? Any kidney problems or stomach ulcers? What's your age and the pain severity (1-10)?"

DISCLAIMER
This information is educational and not a substitute for professional medical advice. If symptoms worsen or you’re worried, seek care. Include this disclaimer when appropriate.
""".strip()

class MediSearchDeepReasonSampler(SamplerBase):
  """
  MediSearch agent using GPT-5 + function/tool calls to query your internal DB
  and synthesize HealthBench-optimized answers.
  """

  def __init__(
    self,
    *,
    model: str = "gpt-5-2025-08-07",
    with_medisearch_db: bool = False,
    max_turns: int = 4,
    max_retries: int = 5,
    temperature: float = 0.2,
    max_parallel_tools: int = 5, # Added for clarity
  ):
    self.model = model
    self.max_turns = max_turns
    self.max_retries = max_retries
    self.temperature = temperature
    self.max_parallel_tools = max_parallel_tools
    self.with_medisearch_db = with_medisearch_db
    self.client = OpenAI()
    logger.info(
        "MediSearchDeepReasonSampler initialized with model '%s', max_turns=%d, max_retries=%d",
        self.model, self.max_turns, self.max_retries
    )

  def _handle_image(
    self,
    image: str,
    encoding: str = "base64",
    format: str = "png",
    fovea: int = 768,
  ):
    new_image = {
      "type": "image_url",
      "image_url": {
        "url": f"data:image/{format};{encoding},{image}",
      },
    }
    return new_image

  def _handle_text(self, text: str):
    return [text]

  def _pack_message(self, role: str, content: Any):
    del role
    return [content]

  def __call__(self, message_list: MessageList) -> SamplerResponse:
    # A unique ID for this specific call to trace logs easily
    call_id = str(uuid4())[:8]
    logger.info("[Call ID: %s] Starting new MediSearch agent call.", call_id)

    tools = [
      {
        "type": "function",
        "function": {
          "name": "search_articles",
          "description": (
            "Search the MediSearch internal articles DB. "
            "Return a JSON array of article objects with keys: "
            "id, title, snip, url, year, journal, n_citations."
          ),
          "parameters": {
            "type": "object",
            "properties": {
              "query": {"type": "string", "description": "Natural language query."},
              "n_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
                "description": "How many results to return."
              }
            },
            "required": ["query"]
          }
        }
      }
    ]


    system_msg = {"role": "developer", "content": HEALTHBENCH_SYSTEM_PROMPT_V2}
    if self.with_medisearch_db:
      messages = [system_msg] + list(message_list)
    else:
      messages = list(message_list)

    trial = 0
    turns = 0
    last_usage = None

    async def _exec_tool(tc):
      # --- Configuration for retry logic ---
      MAX_RETRIES = 4
      INITIAL_BACKOFF = 1  # Start with a 1-second delay

      # Assuming call_id is accessible in this scope
      call_id = getattr(tc, 'id', 'N/A')

      args_raw = tc.function.arguments or "{}"
      logger.info(
          "[Call ID: %s] Executing tool '%s' with raw args: %s",
          call_id, tc.function.name, args_raw
      )
      try:
          try:
              args = json.loads(args_raw)
          except json.JSONDecodeError:
              logger.warning(
                  "[Call ID: %s] JSONDecodeError for tool '%s' args. Treating as string query.",
                  call_id, tc.function.name
              )
              args = {"query": str(args_raw)}

          query = args.get("query", "")
          n_results = int(args.get("n_results", 10))

          # --- Retry logic starts here ---
          last_exception = None
          for attempt in range(MAX_RETRIES):
              try:
                  # Attempt the async DB search
                  result = await search_articles(query, n_results)
                  logger.info(
                      "[Call ID: %s] Tool '%s' search successful on attempt %d.",
                      call_id, tc.function.name, attempt + 1
                  )
                  break  # If successful, exit the retry loop
              except Exception as e:
                  last_exception = e
                  logger.warning(
                      "[Call ID: %s] Attempt %d/%d for tool '%s' failed: %s",
                      call_id, attempt + 1, MAX_RETRIES, tc.function.name, e
                  )
                  if attempt < MAX_RETRIES - 1:
                      # Calculate backoff time: 1s, 2s, 4s, etc. + random jitter
                      backoff_time = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                      logger.info(
                          "[Call ID: %s] Retrying in %.2f seconds...", call_id, backoff_time
                      )
                      await asyncio.sleep(backoff_time)
                  else:
                      # All retries have failed
                      logger.error(
                          "[Call ID: %s] All %d retries failed for tool '%s'. Last error: %s",
                          call_id, MAX_RETRIES, tc.function.name, last_exception, exc_info=True
                      )
                      result = f"No results found for query: {query}"
          # --- Retry logic ends here ---
          logger.debug("[Call ID: %s] Tool '%s' completed successfully.", call_id, tc.function.name)
      except Exception as e:
          logger.error(
              "[Call ID: %s] Error executing tool '%s': %s",
              call_id, tc.function.name, e, exc_info=True
          )
          result = f"ERROR: Could not execute tool {tc.function.name}. Reason: {e}"
      return tc.id, tc.function.name, result


    async def _run_tools(tool_calls):
      calls = list(tool_calls)
      results = []
      total = len(calls)
      if total == 0:
        return results
      for i in range(0, total, self.max_parallel_tools):
        batch = calls[i:i + self.max_parallel_tools]
        logger.info(
          "[Call ID: %s] Running tool batch %d-%d of %d.",
          call_id, i + 1, i + len(batch), total
        )
        tasks = [asyncio.create_task(_exec_tool(tc)) for tc in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=False)
        results.extend(batch_results)
      return results

    while True:
      logger.info(
          "[Call ID: %s] Starting turn %d (API attempt %d)...",
          call_id, turns + 1, trial + 1
      )
      try:
        logger.debug(
            "[Call ID: %s] Calling OpenAI API with %d messages.", call_id, len(messages)
        )
        if self.with_medisearch_db:
          resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            reasoning_effort="high",
            verbosity="high",
          )
        else:
          resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            reasoning_effort="high",
            verbosity="high",
          )
        last_usage = resp.usage
        choice = resp.choices[0]
        msg = choice.message
        logger.info(
            "[Call ID: %s] API call successful. Finish reason: '%s'.",
            call_id, choice.finish_reason
        )

        # Reset trial count on successful API call
        trial = 0

        if getattr(msg, "tool_calls", None):
          logger.info(
              "[Call ID: %s] Model requested %d tool call(s).",
              call_id, len(msg.tool_calls)
          )
          messages.append(msg.model_dump(exclude_unset=True))

          # Run search_articles calls concurrently (async) and feed results back
          results = asyncio.run(_run_tools(msg.tool_calls))
          logger.info(
              "[Call ID: %s] Finished executing %d tool(s).", call_id, len(results)
          )
          for tool_call_id, tool_name, content in results:
            messages.append({
              "role": "tool",
              "tool_call_id": tool_call_id,
              "name": tool_name,
              "content": content,
            })

          turns += 1
          if turns >= self.max_turns:
            logger.warning(
                "[Call ID: %s] Reached max_turns limit (%d). Stopping.",
                call_id, self.max_turns
            )
            messages.append({
              "role": "developer",
              "content": "Finalize now from the context above. Do not call tools."
            })
            resp = self.client.chat.completions.create(
              model=self.model,
              messages=messages,
              reasoning_effort="high",
            )
            return SamplerResponse(
              response_text=resp.choices[0].message.content or "",
              response_metadata={"usage": resp.usage, "call_id": call_id},
              actual_queried_message_list=messages,
            )
          continue # Go to the next turn

        final_text = msg.content or ""
        logger.info("[Call ID: %s] Completed successfully. Returning final response.", call_id)
        return SamplerResponse(
          response_text=final_text,
          response_metadata={"usage": last_usage, "call_id": call_id},
          actual_queried_message_list=messages,
        )

      except Exception as e:
        trial += 1
        wait = 2 ** trial
        logger.error(
            "[Call ID: %s] Exception on turn %d (attempt %d): %s. Retrying in %d seconds...",
            call_id, turns + 1, trial, e, wait, exc_info=True
        )
        if trial >= self.max_retries:
          logger.critical(
              "[Call ID: %s] Failed after %d retries. Aborting call.",
              call_id, self.max_retries
          )
          return SamplerResponse(
            response_text=f"Failed after retries: {e}",
            response_metadata={"usage": None, "call_id": call_id},
            actual_queried_message_list=messages,
          )
        time.sleep(wait)