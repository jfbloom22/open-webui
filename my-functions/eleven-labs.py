"""
title: ElevenLabs TTS Action
author: justinh-rahb
author_url: https://github.com/justinh-rahb
funding_url: https://github.com/open-webui
version: 0.1.1
license: MIT
icon_url: data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3QgeD0iOCIgeT0iNiIgd2lkdGg9IjYiIGhlaWdodD0iMjAiIGZpbGw9IiM0QzRDNEMiLz48cmVjdCB4PSIxOCIgeT0iNiIgd2lkdGg9IjYiIGhlaWdodD0iMjAiIGZpbGw9IiM0QzRDNEMiLz48L3N2Zz4=
required_open_webui_version: 0.3.10
"""

import requests
import uuid
import os
import io
import base64
from pydantic import BaseModel, Field
from typing import Callable, Union, Any, Dict, Tuple
from config import UPLOAD_DIR
from open_webui.models.files import Files, FileForm
from open_webui.storage.provider import Storage

DEBUG = True


class Action:
    class Valves(BaseModel):
        ELEVENLABS_API_KEY: str = Field(
            default=None, description="Your ElevenLabs API key."
        )
        ELEVENLABS_MODEL_ID: str = Field(
            default="eleven_multilingual_v2",
            description="ID of the ElevenLabs TTS model to use.",
        )
        VOICE_NAME: str = Field(
            default="Sarah",
            description="Name of the ElevenLabs voice to use (e.g., Sarah, Rachel, Domi, Bella, Antoni, Elli, Josh, Arnold, Adam, Sam).",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.voice_id_cache = {}

    def status_object(
        self,
        description: str = "Unknown State",
        status: str = "in_progress",
        done: bool = False,
    ) -> Dict:
        return {
            "type": "status",
            "data": {
                "status": status,
                "description": description,
                "done": done,
            },
        }

    async def fetch_available_voices(self) -> Tuple[str, Dict[str, str]]:
        if DEBUG:
            print("Debug: Fetching available voices")

        base_url = "https://api.elevenlabs.io/v1"
        headers = {
            "xi-api-key": self.valves.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }

        voices_url = f"{base_url}/voices"
        try:
            response = requests.get(voices_url, headers=headers)
            response.raise_for_status()
            voices_data = response.json()

            display_message = "Available voices from ElevenLabs:\n\n"
            voice_options = {}
            for voice in voices_data.get("voices", []):
                voice_name = voice["name"]
                voice_id = voice["voice_id"]
                display_message += f"- {voice_name}\n"
                voice_options[voice_name] = voice_id

            if DEBUG:
                print(f"Debug: Found {len(voices_data.get('voices', []))} voices")

            return display_message, voice_options

        except requests.RequestException as e:
            if DEBUG:
                print(f"Debug: Error fetching voices: {str(e)}")
            return "Sorry, couldn't fetch available voices at the moment.", {}

    async def action(
        self,
        body: dict,
        __user__: dict = {},
        __event_emitter__: Callable[[dict], Any] = None,
        __event_call__: Callable[[dict], Any] = None,
    ) -> None:
        if DEBUG:
            print(f"Debug: ElevenLabs TTS action invoked")

        try:
            if __event_emitter__:
                await __event_emitter__(
                    self.status_object("Initializing ElevenLabs Text-to-Speech")
                )

            if not self.valves.ELEVENLABS_API_KEY:
                raise ValueError("ElevenLabs API key is not set")

            if "id" not in __user__:
                raise ValueError("User not authenticated")

            display_message, self.voice_id_cache = await self.fetch_available_voices()

            if not self.voice_id_cache:
                raise ValueError("No available voices to select")

            # Use configured voice from valves
            selected_voice_name = self.valves.VOICE_NAME
            selected_voice_id = self.voice_id_cache.get(selected_voice_name)

            if DEBUG:
                print(
                    f"Debug: Using configured voice: {selected_voice_name} ({selected_voice_id})"
                )

            if not selected_voice_id:
                # Fallback to first available voice if configured voice is not found
                selected_voice_name = list(self.voice_id_cache.keys())[0]
                selected_voice_id = self.voice_id_cache.get(selected_voice_name)
                if DEBUG:
                    print(
                        f"Debug: {self.valves.VOICE_NAME} not found, using fallback voice: {selected_voice_name} ({selected_voice_id})"
                    )

            messages = body.get("messages", [])
            assistant_message = next(
                (
                    message.get("content")
                    for message in reversed(messages)
                    if message.get("role") == "assistant"
                ),
                None,
            )

            if not assistant_message:
                raise ValueError("No assistant message to convert")

            if __event_emitter__:
                await __event_emitter__(self.status_object("Generating speech"))

            base_url = "https://api.elevenlabs.io/v1"
            headers = {
                "xi-api-key": self.valves.ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            }

            tts_url = f"{base_url}/text-to-speech/{selected_voice_id}"
            payload = {
                "text": assistant_message,
                "model_id": self.valves.ELEVENLABS_MODEL_ID,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
            }

            response = requests.post(tts_url, json=payload, headers=headers)
            response.raise_for_status()

            if response.status_code == 200:
                audio_data = response.content
                file_name = f"tts_{uuid.uuid4()}.mp3"

                file_id = self._create_file(
                    file_name, "Generated Audio", audio_data, "audio/mpeg", __user__
                )
                if file_id:
                    file_url = self._get_file_url(file_id, file_name)
                    if __event_emitter__:
                        await __event_emitter__(
                            self.status_object(
                                "Generated successfully", status="complete", done=True
                            )
                        )
                    if file_url:
                        # Create base64 data URL for iOS listening compatibility
                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                        data_url = f"data:audio/mpeg;base64,{audio_base64}"
                        
                        await __event_emitter__(
                            {
                                "type": "message",
                                "data": {
                                    "content": f"""
🎵 **Audio Generated Successfully!**

[📥 Download Audio File]({file_url})

[🎧 Listen to Audio]({data_url})

*Note: Downloading is only supported on desktop browsers. Mobile users can listen directly by tapping the link above.*

"""
                                },
                            }
                        )
                else:
                    raise ValueError("Error saving audio file")
            else:
                raise ValueError(f"Unexpected API response: {response.text}")

        except Exception as e:
            if DEBUG:
                print(f"Debug: Error in action method: {str(e)}")
            if __event_emitter__:
                await __event_emitter__(
                    self.status_object(f"Error: {str(e)}", status="error", done=True)
                )

    def _create_file(
        self,
        file_name: str,
        title: str,
        content: Union[str, bytes],
        content_type: str,
        __user__: dict = {},
    ) -> str:
        if DEBUG:
            print(f"Debug: Creating file: {file_name}")

        if "id" not in __user__:
            if DEBUG:
                print("Debug: User ID is not available")
            return None

        try:
            file_id = str(uuid.uuid4())
            filename_with_id = f"{file_id}_{file_name}"
            
            if isinstance(content, str):
                file_obj = io.StringIO(content)
            else:
                file_obj = io.BytesIO(content)

            contents, file_path = Storage.upload_file(file_obj, filename_with_id, {})

            file_item = Files.insert_new_file(
                __user__["id"],
                FileForm(
                    **{
                        "id": file_id,
                        "filename": file_name,
                        "path": file_path,
                        "meta": {
                            "name": file_name,
                            "content_type": content_type,
                            "size": len(contents),
                            "data": {"title": title},
                        },
                    }
                ),
            )

            if DEBUG:
                print(f"Debug: File saved. Path: {file_path}")
            return file_item.id
        except Exception as e:
            if DEBUG:
                print(f"Debug: Error saving file: {e}")
            return None

    def _get_file_url(self, file_id: str, file_name: str) -> str:
        return f"/api/v1/files/{file_id}/content/{file_name}"
