"""
title: Huggingface Pipe with Chat Support
author: Wes Caldwell
email: Musicheardworldwide@gmail.com
date: 2024-07-19
version: 1.10
license: MIT
description: Function to use HF models on Open-WebUI with chat history support
requirements: pydantic, huggingface-hub, transformers
"""

from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator, List, Dict, Any
import os
import logging
import json
import time
import uuid

from huggingface_hub import HfApi
from transformers import pipeline, set_seed, AutoTokenizer, PipelineException

logging.basicConfig(level=logging.INFO)

class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="HUGGINGFACE/",
            description="Prefix to be added before model names.",
        )
        HUGGINGFACE_API_URL: str = Field(
            default="https://api-inference.huggingface.co/models/",
            description="Base URL for accessing Hugging Face API endpoints.",
        )
        HUGGINGFACE_API_KEY: str = Field(
            default=os.getenv("HUGGINGFACE_API_KEY", ""),
            description="API key for authenticating requests to the Hugging Face API.",
        )
        SYSTEM_PROMPT: str = Field(
            default="You are a helpful assistant.",
            description="Default system message to use when not provided.",
        )
        MAX_HISTORY_LENGTH: int = Field(
            default=10,
            description="Maximum number of messages to include in history.",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.hf_api = HfApi()

    def fetch_models(self) -> List[dict]:
        """Fetch models from Hugging Face containing 'gpt' in their ID."""
        if not self.valves.HUGGINGFACE_API_KEY:
            logging.error("API Key not provided.")
            return [
                {"id": "error", "name": "API Key not provided."}
            ]

        try:
            models = self.hf_api.list_models(
                use_auth_token=self.valves.HUGGINGFACE_API_KEY
            )
            filtered_models = [
                {"id": model.modelId, "name": f"{self.valves.NAME_PREFIX}{model.modelId}"}
                for model in models
                if "gpt" in model.modelId
            ]
            return filtered_models

        except Exception as e:
            logging.error(f"Failed to fetch models: {e}")
            return [
                {"id": "error", "name": "Could not fetch models. Update the API Key."}
            ]

    def _format_messages_to_prompt(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> str:
        """Convert a list of chat messages to a single prompt string."""
        prompt = ""
        
        # Add system prompt if provided
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"
        elif self.valves.SYSTEM_PROMPT:
            prompt += f"System: {self.valves.SYSTEM_PROMPT}\n\n"
            
        # Add message history
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
            elif role == 'system':
                # System messages already handled at the beginning
                continue
                
        # Add final assistant prefix to prompt for the model to continue
        prompt += "Assistant: "
        
        return prompt

    def _create_openai_response(self, 
                               model_id: str, 
                               content: str, 
                               stream: bool = False,
                               is_final: bool = False) -> Dict[str, Any]:
        """Create an OpenAI-compatible response format."""
        timestamp = int(time.time())
        message_id = f"{model_id}-{str(uuid.uuid4())}"
        
        if stream and not is_final:
            # For streaming responses (chunks)
            return {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "logprobs": None,
                        "finish_reason": None
                    }
                ]
            }
        elif stream and is_final:
            # Final streaming chunk
            return {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ]
            }
        else:
            # Non-streaming complete response
            return {
                "id": message_id,
                "object": "chat.completion",
                "created": timestamp,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # These would need actual token counts
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

    def pipe(self, body: dict, __user__: dict) -> Union[Dict[str, Any], Generator, Iterator]:
        """Process a chat completion request with message history support."""
        if "model" not in body:
            logging.error("Model not specified in the request body.")
            return {"error": "Model not specified in the request body."}

        model_id = body["model"]
        logging.info(f"Processing request for model: {model_id}")
        
        # Extract messages or fall back to prompt
        messages = body.get("messages", [])
        if not messages and "prompt" in body:
            # Legacy support for old prompt format
            messages = [{"role": "user", "content": body["prompt"]}]
            
        if not messages:
            logging.error("No messages or prompt provided in the request body.")
            return {"error": "No messages or prompt provided."}
            
        # Get system prompt if provided in options
        system_prompt = None
        if "options" in body and isinstance(body["options"], dict):
            system_prompt = body["options"].get("system")
            
        # Extract other parameters
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 100)
        if "options" in body and isinstance(body["options"], dict):
            # Support for Ollama-style options
            if "num_predict" in body["options"]:
                max_tokens = body["options"]["num_predict"]
        
        try:
            # Format messages into a prompt
            formatted_prompt = self._format_messages_to_prompt(messages, system_prompt)
            logging.info(f"Formatted prompt: {formatted_prompt}")
            
            # Set up the text generation pipeline
            device = 0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            generator = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=tokenizer,
                device=device,
            )
            
            set_seed(42)  # Optional: Ensure reproducibility

            # Generate response
            if stream:
                def streaming_generator():
                    completion = ""
                    response = generator(
                        formatted_prompt,
                        max_length=len(tokenizer.encode(formatted_prompt)) + max_tokens,
                        num_return_sequences=1,
                        do_sample=True,
                        return_full_text=False,
                    )
                    
                    # Get the generated text
                    generated_text = response[0]["generated_text"]
                    
                    # Stream character by character
                    for char in generated_text:
                        completion += char
                        yield f"data: {json.dumps(self._create_openai_response(model_id, char, stream=True))}\n\n"
                    
                    # Send final [DONE] message
                    yield f"data: {json.dumps(self._create_openai_response(model_id, '', stream=True, is_final=True))}\n\n"
                    yield "data: [DONE]\n\n"
                
                return streaming_generator()
            else:
                # Non-streaming response
                response = generator(
                    formatted_prompt,
                    max_length=len(tokenizer.encode(formatted_prompt)) + max_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    return_full_text=False,
                )
                
                generated_text = response[0]["generated_text"]
                return self._create_openai_response(model_id, generated_text)

        except PipelineException as pe:
            logging.error(f"Pipeline error: {pe}")
            return {"error": f"Pipeline error: {pe}"}
        except Exception as e:
            logging.error(f"Failed to process request: {e}")
            return {"error": f"Failed to process request: {e}"}

if __name__ == "__main__":
    pipe = Pipe()
    
    # Test fetch_models
    models = pipe.fetch_models()
    print("Available models:", models)
    
    # Test with a single message (legacy mode)
    test_body_legacy = {
        "model": "gpt2",
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "stream": False,
    }
    result_legacy = pipe.pipe(test_body_legacy, __user__={})
    print("\nLegacy mode response:")
    print(json.dumps(result_legacy, indent=2))
    
    # Test with chat history
    test_body_chat = {
        "model": "gpt2",
        "messages": [
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": "Tell me a story about a dragon."},
            {"role": "assistant", "content": "Once upon a time, there lived a mighty dragon named Infernus."},
            {"role": "user", "content": "What happened next?"}
        ],
        "max_tokens": 50,
        "stream": False,
    }
    result_chat = pipe.pipe(test_body_chat, __user__={})
    print("\nChat mode response:")
    print(json.dumps(result_chat, indent=2)) 