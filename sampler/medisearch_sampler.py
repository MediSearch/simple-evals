import time
from typing import Any

from medisearch_client import MediSearchClient, Settings
import uuid

from ..eval_types import MessageList, SamplerBase, SamplerResponse


class MediSearchSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API for o series models
    """

    def __init__(
        self,
        *,
        model: str = "pro",
    ):
        self.client = MediSearchClient(api_key="0b146290-ebcb-48d0-aec1-f2fa0cbc4893", 
                                       base_url="https://api.backend.medisearch.io")
        self.model = model

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
    
    def _transform_to_medisearch_messages(
            self, message_list: MessageList) -> list[dict]:
      """In HealthBench the roles may come in the format:
      # USER, ASSISTANT, USER

      but also USER, ASSISTANT, ASSISTANT, USER

      We should merge consecutive messages with the same role, and finally
      make sure that our format is USER, ASSISTANT, USER, ASSISTANT, USER, etc.
      """
      medisearch_messages = []
      current_role = None
      print(message_list)
      for message in message_list:
          if message["role"] == current_role:
              medisearch_messages[-1] += "\n" + message["content"]
          else:
              medisearch_messages.append(message["content"])
              current_role = message["role"]
      return medisearch_messages

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        messages = self._transform_to_medisearch_messages(message_list)
        while True:
            try:
                responses = self.client.send_message(
                    conversation=messages,
                    conversation_id=str(uuid.uuid4()),
                    settings=Settings(
                        model_type=self.model,
                    )

                )
                response_text = ""
                for response in responses:
                  if response["event"] == "llm_response":
                    response_text = response["data"]
                    break
                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                print("MediSearchSampler error")
                print(e)
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
