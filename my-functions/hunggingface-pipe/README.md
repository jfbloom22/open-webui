# Huggingface Pipe with Chat Support

This function allows you to use Hugging Face models directly in Open-WebUI with full support for chat history and conversation context.

## Features

- **Chat History Support**: Maintains conversation context across multiple messages
- **OpenAI-Compatible Interface**: Returns responses in OpenAI format for seamless integration
- **Streaming Support**: Works with both streaming and non-streaming requests
- **System Prompt Support**: Configurable system prompt
- **Backward Compatibility**: Still supports the original single-prompt format

## Installation

1. Make sure you have the required dependencies:
   ```bash
   pip install pydantic huggingface-hub transformers
   ```

2. Set your Hugging Face API key as an environment variable:
   ```bash
   export HUGGINGFACE_API_KEY="your_api_key"
   ```

3. Copy this function to your Open-WebUI functions directory.

## Configuration Options

The function has several configurable parameters in the `Valves` class:

- `NAME_PREFIX`: Prefix added to model names (default: "HUGGINGFACE/")
- `HUGGINGFACE_API_URL`: Base URL for Hugging Face API endpoints
- `HUGGINGFACE_API_KEY`: API key for authentication (defaults to environment variable)
- `SYSTEM_PROMPT`: Default system message when not provided
- `MAX_HISTORY_LENGTH`: Maximum number of messages to include in history

## Usage Examples

### Basic Chat Example

```python
from functions.huggingface_pipe import Pipe

pipe = Pipe()

# Chat with history
response = pipe.pipe({
    "model": "gpt2",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I'm an AI assistant powered by gpt2."},
        {"role": "user", "content": "What can you do?"}
    ],
    "max_tokens": 100,
    "stream": False
}, __user__={})

print(response)
```

### Legacy Mode Example

```python
# Single prompt (legacy mode)
response = pipe.pipe({
    "model": "gpt2",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "stream": False
}, __user__={})

print(response)
```

### Streaming Example

```python
# Streaming response
stream = pipe.pipe({
    "model": "gpt2",
    "messages": [
        {"role": "user", "content": "Write a short poem."}
    ],
    "max_tokens": 100,
    "stream": True
}, __user__={})

for chunk in stream:
    print(chunk)
```

## How It Works

1. The function converts the message history into a formatted prompt
2. It uses the Hugging Face Transformers library to generate responses
3. The response is formatted in OpenAI-compatible format
4. For streaming responses, it yields chunks in SSE format

## Customization

You can customize the prompt format by modifying the `_format_messages_to_prompt` method, which currently formats messages as:

```
System: {system_prompt}

User: {message1}
Assistant: {message2}
User: {message3}
Assistant: 
``` 