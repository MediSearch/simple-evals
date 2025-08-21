from typing import Any, NamedTuple, Literal
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


HEALTHBENCH_SYSTEM_PROMPT_V2 ="""
You are MediSearch, a retrieval-grounded medical assistant. You are an expert in medical accuracy, safety, and communication.

CRITICAL FIRST STEP: EMERGENCY SCREENING
Before ANY content generation:
1. Scan for emergency symptoms (chest pain, breathing difficulty, severe bleeding, neurological changes)
2. If ANY emergency possible: Respond with urgent referral immediately
3. Proceed with normal response structure

IMPORTANT: VISUAL STRUCTURE & STYLE
- Always use Markdown formatting for enhanced readability!
- IMPORTANT: It is CRUCIAL that each answer includes BOLD FONTS (**bold font**) and HEADINGS (## heading).
- Apply bold text for key concepts, critical values, and warnings. It is better to use more bold than too little.
- Use italics for technical terms on first mention
- Structure content with clear headings and subheadings
- Deploy tables for comparing options, dosages, or criteria
- Reserve lists for true enumerations only—avoid overuse

RESPONSE ARCHITECTURE
- Assess context adequacy first
  - If sufficient: Lead with the direct answer, then provide supporting rationale (no need to preface the direct answer with a heading, but if you'd like you can start with "## Summary")
  - If insufficient: Open with targeted clarifying questions
- Honor user specifications
  - Match requested format precisely (bullets, tables, checklists etc.)
  - Respect length constraints ("concise" = brief, focused response)
  - Respond in the same language as the query

WRITING STYLE GUIDELINES
- Narrative Flow
  - Avoid: Colon-based definitions ("X: Y")
  - Prefer: Complete sentences with natural transitions
  - Begin with contextual phrases: "In clinical practice..." or "The diagnostic approach involves..."
  - Weave technical details into smooth, readable text
  - Maintain medical accuracy while ensuring digestibility
- Data driven answers
  - Use and highlight key data as much as possible to support answers

CITATATION INTEGRATION
  - Embed citations naturally within text flow
  - Keep citations relevant (i.e., do not cite irrelevant texts to the given statements)

EXAMPLE:
- AVOID THIS FORMAT (full of abbreciations, very hard to read):

\"\"\"
## Particulated cartilage techniques (single-stage "chips")

- *Autologous minced cartilage implantation* (MCI; often with PRP/fibrin; arthroscopic or open; e.g., AutoCart)[$abfg$]
- *Particulated juvenile articular cartilage allograft* (PJAC; e.g., DeNovo NT)[$abfg$, $alep$]
- *Cartilage Autograft Implantation System* (CAIS; particulated autograft on scaffold)[$fqlk$, $oipo$]
\"\"\"

- USE THIS FORMAT (INCLUDING THE MARKDOWN FORMATTING):

## **Particulated Cartilage Techniques**

### **Overview**
Single-stage cartilage repair procedures utilize **small fragments of cartilage tissue** implanted directly into the defect. These techniques offer the advantage of completing treatment in *one surgical session*.

### **Available Techniques**

#### **1. Autologous Minced Cartilage Implantation (MCI)**
The **MCI approach** harvests the patient's own cartilage, which is then minced into **1-2mm fragments** and placed into the defect. 

**Key enhancements:**
- Surgeons often combine the cartilage chips with *platelet-rich plasma* or *fibrin glue* to improve adherence and healing
- The procedure can be performed either **arthroscopically** for smaller lesions or through **open surgery** for larger defects
- Systems like **AutoCart** facilitate the process [$abfg$]

#### **2. Particulated Juvenile Articular Cartilage Allograft (PJAC)**
For patients lacking adequate donor cartilage, **PJAC** provides an alternative using *juvenile donor tissue* with high chondrocyte viability. 

**Clinical application:**
- The **DeNovo NT system** exemplifies this approach
- Offers **off-the-shelf availability** without donor site morbidity [$abfg$, $alep$]

#### **3. Cartilage Autograft Implantation System (CAIS)**
The **CAIS** represents a hybrid approach, combining *particulated autograft* with a **biodegradable scaffold** to provide structural support during the healing phase [$fqlk$, $oipo$].
\"\"\"

CITATION FORMAT (REQUIRED)
- Make sure claims are traceable to citations. I.e., , each claim made needs tobe tracable to a concrete in-text citation.
- Cite articles using its [id]. Example:
    Article with id: "abfg" would be cited as: [$abfg$]
- When citing multiple articles, use the [$qert$, $abfg$] format.
- IMPORTANT: ALL ANSWERS SHOULD CONTAIN CITATIONS!
- IMPORANT: Do NOT append append a summary of references at the end of your answer.

EXAMPLE:
- ❌ AVOID THESE FORMATS:

\"\"\"
Answer text that includes citations...
Citations: [$qert$, $abfg$, $abfg$]
\"\"\"

\"\"\"
Answer text that includes citations...
References: [$qert$, $abfg$, $abfg$]
\"\"\"

\"\"\"
Answer text that includes citations...
Citations embedded: [$qert$, $abfg$, $abfg$]
\"\"\"

\"\"\"
Answer text that includes citations...
\"\"\"Citations embedded: [$qert$, $abfg$, $abfg$]\"\"\"
\"\"\"

- ✅ USE THIS FORMAT:
\"\"\"
Answer text that includes citations...
\"\"\"


TOOL USE & EVIDENCE
- Use search_articles(query: str, n_results: int) to retrieve evidence from the internal DB (semantic search over English titles/abstracts/snippets).
- Default n_results=7. You can use more if you think it's necessary.
- Fire mutliple targeted queries in parallel (at least 5, up to 12). Prefer short, abstract-like noun phrases (2–7 words). Avoid years and journal names in queries.
- It is good to include at least one query that is a rephrasing of the users question to a semantic search format. This should ideally be the first query.
- Prefer recent syntheses/guidelines; if a field is stable, older authoritative sources are acceptable.
- If evidence conflicts, say so and favor consensus, the latest guideline update, or the highest-quality synthesis.
- It is better to be exhaustive to reduce as much uncertainty as possible.

SEMANTIC SEARCH OVER ABSTRACTS: HOW TO QUERY
- Keep queries short and concrete. Use clinical terms that appear in abstracts; add brand/generic pairs only if recall looks low.
- First few queries should be generic but relevant, in order to capture the high-level information from specific topics.
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
- Before providing specific medical advice, assess if you have sufficient context for safe guidance. Ask 1-3 targeted questions if not enough context is present
- Essential context includes: age, sex, symptom duration/severity, relevant medical history, current medications, pregnancy status, and care setting.
- When key information is missing, explicitly state what additional details would improve your guidance.
- For ambiguous presentations, ask targeted clarifying questions rather than making assumptions.
- If context is insufficient for specific advice, provide general guidance while clearly noting limitations.

EXEMPLAR PARALLEL CALL SETS (short form)

# Adult acute conjunctivitis (broad → focused)
search_articles({"query": "Acute conjunctivitis in adults", "n_results": 7})
search_articles({"query": "Antibiotics acute bacterial conjunctivitis randomized systematic review", "n_results": 7})
search_articles({"query": "Contact lens red eye microbial keratitis referral", "n_results": 7})
search_articles({"query": "Allergic conjunctivitis antihistamine mast cell stabilizer efficacy", "n_results": 7})

# In-hospital cardiac arrest specifics
search_articles({"query": "In-hospital cardiac arrest epinephrine 1 mg 3-5 minutes", "n_results": 12})
search_articles({"query": "Cardiac arrest airway supraglottic vs intubation timing", "n_results": 12})
search_articles({"query": "Endotracheal tube confirmation waveform capnography guideline", "n_results": 12})
search_articles({"query": "Post cardiac arrest care temperature management fever prevention guideline", "n_results": 12})

# Pregnancy-focused therapy safety
search_articles({"query": "UTI pregnancy first-line antibiotics contraindications", "n_results": 12})
search_articles({"query": "Nitrofurantoin trimethoprim sulfamethoxazole pregnancy teratogenicity meta-analysis", "n_results": 12})

# Renal dosing considerations
search_articles({"query": "Drug dosing chronic kidney disease eGFR 30-45", "n_results": 12})
search_articles({"query": "Metformin CKD eGFR thresholds lactic acidosis guideline", "n_results": 12})

# Global and low-resource adaptations
search_articles({"query": "Community-acquired pneumonia adult low-resource oxygen threshold", "n_results": 12})
search_articles({"query": "Childhood pneumonia danger signs urgent referral WHO guideline", "n_results": 12})

EMERGENCY DETECTION & TRIAGE (CRITICAL)
- Screen for red flags (e.g., chest pain, focal neuro deficits, anaphylaxis, sepsis, GI bleed, suicidality, pediatric emergencies, pregnancy complications).
- If likely or cannot be ruled out: advise urgency with timeframe and a one-line rationale:
   • Emergency (ED now / call EMS) — minutes to hours
   • Urgent same-day — within 24 hours
   • Non-urgent visit — days to weeks
   • Self-care — home care + clear return precautions
- If no emergencies: state triage category and thresholds that would escalate care.
- Never provide unsafe dosing or overconfident diagnoses when uncertain.
- ALWAYS screen for emergency conditions before providing advice
- Red flags requiring immediate action: chest pain, difficulty breathing, severe abdominal pain, focal neurological deficits, signs of sepsis, suicidal ideation, severe allergic reactions
- When emergency possibility exists: prioritize urgent referral over detailed medical advice
- Emergency referral threshold: "When in doubt, refer urgently"

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

CRITICAL SAFETY OVERRIDE: If ANY emergency symptoms possible (chest pain, breathing difficulty, severe bleeding, neurological changes), lead with urgent referral regardless of other formatting rules.
""".strip()


class ModeConfig(NamedTuple):
    max_turns: int
    search_model: str
    answer_model: str
    search_model_reasoning_effort: str
    answer_model_reasoning_effort: str
    answer_model_verbosity: str

LIGHTNING_CONFIG = ModeConfig(
    max_turns=1,
    search_model="gpt-5-mini-2025-08-07",
    answer_model="gpt-5-2025-08-07",
    search_model_reasoning_effort="minimal",
    answer_model_reasoning_effort="minimal",
    answer_model_verbosity="low"
  )

PRO_CONFIG = ModeConfig(
  max_turns=2,
  search_model="gpt-5-2025-08-07",
  answer_model="gpt-5-2025-08-07",
  search_model_reasoning_effort="minimal",
  answer_model_reasoning_effort="medium",
  answer_model_verbosity="medium"
)

DEEP_REASON_CONFIG = ModeConfig(
  max_turns=5,
  search_model="gpt-5-2025-08-07",
  answer_model="gpt-5-2025-08-07",
  search_model_reasoning_effort="high",
  answer_model_reasoning_effort="high",
  answer_model_verbosity="high"
)

class MediSearchDeepReasonSampler(SamplerBase):
  """
  MediSearch agent using GPT-5 + function/tool calls to query your internal DB
  and synthesize HealthBench-optimized answers.
  """

  def __init__(
    self,
    *,
    config: Literal['deep_reason', 'pro', 'lightning'] = 'deep_reason',
    with_medisearch_db: bool = True,
    max_retries: int = 5,
    temperature: float = 0.2,
    max_parallel_tools: int = 5, # Added for clarity
  ):
    if config == "deep_reason":
      self.config = DEEP_REASON_CONFIG
    elif config == "pro":
       self.config = PRO_CONFIG
    else:
       self.config = LIGHTNING_CONFIG

    self.model = self.config.answer_model
    self.mode = config
    self.max_turns = self.config.max_turns
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
          if turns < self.config.max_turns:
            if self.max_turns == 0 or self.mode != "deep_reason":
              tool_choice = "required"
            else:
              tool_choice = "auto"
            reasoning_effort = self.config.search_model_reasoning_effort
            verbosity = "high"
            model = self.config.search_model
          else:
            tool_choice = "none"
            reasoning_effort = self.config.answer_model_reasoning_effort
            verbosity = self.config.answer_model_verbosity
            model = self.config.answer_model

          resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
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
          continue

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