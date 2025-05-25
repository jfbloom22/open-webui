"""
title: SIOP Schedule Search with LLM Enhancement
description: Searches SIOP conference sessions using ChromaDB and LLM processing
version: 0.4.0
requirements: pydantic>=1.10.0,chromadb
"""

import logging
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Generator, Union, Iterator
import chromadb
from open_webui.main import generate_chat_completions
import json
from open_webui.models.users import User

logger = logging.getLogger('siop_search_pipe')
logger.setLevel(logging.INFO)

# Prompt Templates
DATE_EXTRACTION_SYSTEM_PROMPT = """
Extract the conference date from the query. 
Return ONLY the date in YYYY-MM-DD format if found, otherwise return 'none'.
Do not include any explanation or additional text in your response.
"""

DATE_EXTRACTION_USER_PROMPT = """Valid dates are:
Wednesday April 2, 2025 (2025-04-02)
Thursday April 3, 2025 (2025-04-03)
Friday April 4, 2025 (2025-04-04)
Saturday April 5, 2025 (2025-04-05)

Previous conversation:
{history_context}

Query: "{query}"

Return ONLY ONE of these values: 2025-04-02, 2025-04-03, 2025-04-04, 2025-04-05, or none
"""

QUERY_ENHANCEMENT_SYSTEM_PROMPT = """Create an enhanced search query for finding conference sessions
Include relevant keywords and synonyms related to the query topic.
Do NOT include date terms in the search query as date filtering is handled separately.
Focus on extracting key concepts, skills, topics, and relevant industry terms.
Return only the optimized search text without any explanation.
"""

QUERY_ENHANCEMENT_USER_PROMPT = """Query: {query}

Previous conversation:
{history_context}

Return only the optimized search terms, no explanation.
"""

RESPONSE_GENERATION_SYSTEM_PROMPT = """
## 🧑‍💼 ROLE
You are an I-O expert assistant excited about the **2025 SIOP Annual Conference** in Denver, CO. You help the user:
- Explore the official SIOP schedule
- Discover relevant sessions
- Build a personalized conference plan


You **ONLY** use information from the official schedule documents. No guessing. No hallucinations. One day at a time.


---

## 🎯 OBJECTIVE
Optimize the user's conference experience based on:
- Interests & preferred topics
- Session format preferences
- Speaker or organization interests
- Networking or career goals
- Time constraints
- Backup session options for time slots with multiple relevant sessions


---

### 🗓️ Typical Session Blocks (For Context Only)

These are common session time blocks for planning purposes. **You do not need to fill every block—only suggest sessions that are explicitly listed in the schedule.**

- **April 2 (Wednesday)** – Preconference workshops  
  *(Times vary — not structured into blocks)*


- **April 3 & 4 (Thursday & Friday)**  
  - 08:00  
  - 09:00  
  - 10:30  
  - 13:00  
  - 14:00  
  - 16:00  
  - 17:00


- **April 5 (Saturday)**  
  - 08:00  
  - 09:30  
  - 10:30  
  - 12:30  
  - 14:00  
  - 15:00

### 🔍 Secondary Goal: Time-Based Planning

- When matching sessions, try to **cover these time blocks where possible** based on the user's interests.
- **Do not create or infer sessions** to fill gaps.
- If no session is found for a time block, simply leave it unfilled and offer to search again or explore alternate topics.

⛔ Priority: **Only list sessions explicitly found in the schedule. Never fabricate or assume session details.**

---

## 🧭 HOW TO INTERACT
1. Start with scheduling Thursday ONLY (unless the user specified a **date**)
2. There are a tons of sessions to search through, you can only schedule up to ONE day per response.  Feel free to inform the user about this.
3. For each time slot:
   - Present the most relevant session as the primary option
   - If there are other sessions with relevance score > 80% at the same time:
     - Include up to 2 backup options
     - Label them clearly as "Backup Session"
     - Only show backups that are highly relevant to the search query
4. Order sessions by date and time
5. If no match is found for the topic, date, or time requested:
   - Say so clearly
   - Suggest new keywords or topics
   - Offer to search again
6. If the user asks about a date that is not available, suggest the nearest available date
7. When providing a list of sessions for a day, ask for feedback or if they would like sessions for the next day
8. If the user asks you to schedule more than one day, respond saying you can only schedule one day at a time and to please ask again. (this is a hard rule because there are too many sessions to schedule more than one day at a time)

---

## 🚫 HARD RULES
1. **Never fabricate sessions**
2. Only show sessions explicitly listed in the search results
3. **Do not infer** speaker info, topics, or formats not provided
4. For more info, refer the user to the [SIOP 2025 Schedule](https://www.siop.org/events/the-annual-conference/attendee-info/schedule/)
5. Only answer questions about the SIOP 2025 conference

---

## ✅ SESSION FORMATS (User Preferences May Include These)
- **Debate** – Moderated with opposing views and Q&A
- **IGNITE!** – Fast-paced, 3-5 minute lightning talks
- **Master Tutorial** – Educational, practical, cross-disciplinary
- **Panel Discussion** – 3–5 experts + moderator + interaction
- **Poster Session** – Visual research displays (a type of subsession)

---

## 📌 NOTES
- You're an expert assistant, but limited to the official schedule
- Speak conversationally but be concise
- Keep answers structured, helpful, and easy to scan
- When providing a schedule for Saturday, suggest the user export their schedule by clicking "Export Schedule"
- When generating a summary, include a disclaimer and a promotion:
  - Remind the user that AI may hallucinate and encourage them to **double-check the official schedule** here: [SIOP 2025 Conference Schedule](https://www.siop.org/events/the-annual-conference/attendee-info/schedule/)
  - 👉 **"Want to learn how to build a custom AI Assistant like this? Visit [AI for HR Mastermind](https://www.aiforhrmastermind.com) today!"**
  
### Session Formatting Example:
```
## Thursday April 3rd, 2025  
* **🕒 12:30 PM - 1:50 PM MDT**  
    * **Leadership 4.0: AI-Powered Leadership Assessment and Development - (Session ID: 5566)**  
    * **Location:** 201

    * **Backup Options (Same Time):**
        * **1. Strategic Leadership in Digital Age - (Session ID: 5567)**
        * **Location:** 202

        * **2. Executive Leadership Workshop - (Session ID: 5568)**
        * **Location:** 203

* **🕒 2:00 PM - 2:50 PM MDT**  
    * **AI-Powered Leadership in IO - (Session ID: 2345)**  
    * **Location:** 201  

* **🕒 3:00 PM - 3:50 PM MDT**  
    * **Emergent AI Competencies for Future-Ready Leaders - (Session ID: 3342)**  
    * **Location:** 201  
    * **Speakers:** Dr. Lena Morris, Raj Patel  
    * **Track:** Leadership & AI Integration  
    * **Description:** Explore critical AI-related competencies leaders must develop to remain relevant in rapidly transforming organizations.

    * **Backup Option (Same Time):**
        * **From Insight to Action: Leveraging AI in Leadership Coaching - (Session ID: 7712)**  
        * **Location:** 202  
        * **Speakers:** Jennifer Hsu, Daniel Cortez  
        * **Track:** Applied AI in Organizational Development  
        * **Description:** This session bridges AI-generated insights with real-world coaching interventions, focusing on enhancing leader effectiveness at scale.
```

"""

RESPONSE_GENERATION_USER_PROMPT = """
## Date Filter: {date_context}
<previous_conversation>
{history_context}
</previous_conversation>

## Search results (ordered by relevance):
<search_results>
{results}
</search_results>

Please provide the most relevant sessions from these results.

Respond to the user's question: {query}
"""

class Pipe:
    class Valves(BaseModel):
        COLLECTION_NAME: str = Field(default='siop_sessions')
        RESULTS_LIMIT: int = Field(default=15)
        HOST: str = Field(default='localhost')
        PORT: str = Field(default='8000')
        AUTH_TOKEN: str = Field(
            default='',
            description='Authentication token for ChromaDB server'
        )
        MODEL: str = Field(
            default='gpt-3.5-turbo-1106',
            description='Model to use for LLM calls. Must be a valid model name from your configured providers.'
        )
        TEMPERATURE: float = Field(
            default=0.7,
            description='Temperature for LLM responses (0.0 to 1.0)'
        )
        TENANT: str = Field(
            default='default_tenant',
            description='ChromaDB tenant name'
        )
        DATABASE: str = Field(
            default='default_database',
            description='ChromaDB database name'
        )
        DEBUG: bool = Field(
            default=False,
            description='Enable detailed debug logging'
        )
        
    def __init__(self):
        self.valves = self.Valves()
        self._client = None
        self.conference_dates = {
            'wednesday': '2025-04-02',
            'thursday': '2025-04-03', 
            'friday': '2025-04-04',
            'saturday': '2025-04-05'
        }

    def pipes(self):
        return [{'id': 'siop-llm-search', 'name': 'SIOP Schedule Search'}]

    async def _llm_call(self, system_prompt: str, user_prompt: str, temperature: float, stream: bool, __user__: dict, __request__) -> dict:
        """Call the LLM API with the provided prompts and parameters."""
        if self.valves.DEBUG:
            logger.info(f"Calling LLM with system prompt: '{system_prompt}' and user prompt: '{user_prompt}'")
        else:
            logger.info("Calling LLM API")
        
        # Create properly formatted messages array
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        form_data = {
            "model": self.valves.MODEL,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if self.valves.DEBUG:
            logger.info(f"LLM call form data: {json.dumps(form_data, default=str)[:200]}...")
        
        try:
            response = await generate_chat_completions(
                request=__request__,
                form_data=form_data,
                user=__user__
            )
            return response
        except Exception as e:
            logger.error(f"LLM call failed with error: {str(e)}")
            if self.valves.DEBUG:
                logger.error(f"Request form data: {json.dumps(form_data, default=str)[:200]}...")
            raise

    def _process_chunk(self, chunk):
        """Process a streaming chunk to extract just the content."""
        try:
            if chunk is None:
                return None
            
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode('utf-8')
            else:
                chunk_str = chunk
            
            if chunk_str.startswith('data: '):
                chunk_str = chunk_str[6:]
            
            chunk_str = chunk_str.strip()
            
            if chunk_str == "[DONE]" or not chunk_str:
                return None
            
            try:
                chunk_data = json.loads(chunk_str)
                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                    delta = chunk_data["choices"][0].get("delta", {})
                    if delta is not None and "content" in delta:
                        return delta["content"]
                    # Check for finish reason
                    finish_reason = chunk_data["choices"][0].get("finish_reason")
                    if finish_reason:
                        logger.info(f"Stream finished with reason: {finish_reason}")
                return None
            except json.JSONDecodeError:
                logger.error(f'ChunkDecodeError: unable to parse "{chunk_str[:100]}"')
                return None
            except KeyError as e:
                logger.error(f"Missing key in chunk data: {e}")
                if self.valves.DEBUG:
                    logger.debug(f"Chunk data: {chunk_str[:200]}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error processing chunk: {str(e)}")
            return None

    async def _extract_date_llm(self, query: str, chat_history: List[Dict], __user__: dict, __request__) -> Optional[str]:
        """Extract a conference date from the user query using LLM."""
        system_prompt = DATE_EXTRACTION_SYSTEM_PROMPT
        
        # Format chat history for context
        history_context = ""
        if chat_history and len(chat_history) > 1:
            history_context = ""
            # Include up to 3 previous exchanges but skip the most recent query
            for msg in chat_history[:-1][-6:]:  # Get up to 3 exchanges (6 messages)
                if isinstance(msg, dict) and msg.get('content'):
                    role = "User" if msg.get('role', 'user').lower() == 'user' else "Assistant"
                    content = msg.get('content', '')
                    history_context += f"{role}: {content}\n"
        
        user_prompt = DATE_EXTRACTION_USER_PROMPT.format(
            query=query,
            history_context=history_context
        )
        try:
            response = await self._llm_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                stream=False,
                __user__=__user__,
                __request__=__request__
            )
            
            date = response['choices'][0]['message']['content'].strip()
            
            # Validate that we got a proper date format
            if date.lower() == 'none':
                return None
                
            # Make sure it's one of our valid conference dates
            if date in ['2025-04-02', '2025-04-03', '2025-04-04', '2025-04-05']:
                logger.info(f"Extracted date: {date}")
                return date
            else:
                logger.warning(f"Invalid date format returned by LLM: {date}")
                return None
        except Exception as e:
            logger.error(f"Error in LLM date extraction: {str(e)}")
            return None

    async def _enhance_query_llm(self, query: str, date: Optional[str], chat_history: List[Dict], __user__: dict, __request__) -> str:
        """Enhance the search query using LLM to improve search results."""
        # Format chat history for context
        history_context = ""
        if chat_history and len(chat_history) > 1:
            history_context = ""
            # Include up to 3 previous exchanges but skip the most recent query
            for msg in chat_history[:-1][-6:]:  # Get up to 3 exchanges (6 messages)
                if isinstance(msg, dict) and msg.get('content'):
                    role = "User" if msg.get('role', 'user').lower() == 'user' else "Assistant"
                    content = msg.get('content', '')
                    history_context += f"{role}: {content}\n"
            
        user_prompt = QUERY_ENHANCEMENT_USER_PROMPT.format(
            query=query,
            history_context=history_context
        )
        
        try:
            response = await self._llm_call(
                system_prompt=QUERY_ENHANCEMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
                stream=False,
                __user__=__user__,
                __request__=__request__
            )
            
            enhanced = response['choices'][0]['message']['content'].strip()
            logger.info(f"Enhanced query: '{enhanced}'" + (f" (original: '{query}')" if self.valves.DEBUG else ""))
            return enhanced
        except Exception as e:
            logger.error(f"Error in query enhancement: {str(e)}")
            return query  # Fall back to original query if enhancement fails

    def _get_chromadb_client(self):
        """Get or create ChromaDB client."""
        if not self._client:
            headers = None
            if self.valves.AUTH_TOKEN:
                headers = {
                    'Authorization': f'Bearer {self.valves.AUTH_TOKEN}'
                }
            self._client = chromadb.HttpClient(
                host=self.valves.HOST,
                port=self.valves.PORT,
                headers=headers,
                tenant=self.valves.TENANT,
                database=self.valves.DATABASE
            )
        return self._client

    def _check_collection_exists(self):
        """Check if the ChromaDB collection exists."""
        try:
            client = self._get_chromadb_client()
            client.get_collection(self.valves.COLLECTION_NAME)
            return True
        except ValueError:
            logger.error(f"Collection {self.valves.COLLECTION_NAME} does not exist.")
            return False
        except Exception as e:
            logger.error(f"Error checking collection: {str(e)}")
            return False

    def _search_sessions(self, query: str, date: Optional[str] = None):
        """Search for sessions in ChromaDB."""
        if not self._check_collection_exists():
            return None
            
        client = self._get_chromadb_client()
        collection = client.get_collection(self.valves.COLLECTION_NAME)
        
        where_filter = {'date_iso': {'$eq': date}} if date else None
        
        # Log the search parameters
        search_params = {
            'query_texts': [query],
            'n_results': self.valves.RESULTS_LIMIT,
            'where': where_filter,
            'include': ["documents", "metadatas", "distances"]
        }
        
        if self.valves.DEBUG:
            logger.info(f"ChromaDB search parameters: {json.dumps(search_params, default=str)}")
        else:
            logger.info(f"Searching ChromaDB with query: '{query[:30]}{'...' if len(query) > 30 else ''}'" + (f" for date: {date}" if date else ""))
        
        # Execute the query
        results = collection.query(
            query_texts=[query],
            n_results=self.valves.RESULTS_LIMIT,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def _format_results(self, results: Dict) -> str:
        """Format ChromaDB results into a structured text format."""
        if not results or not results.get('ids') or not results['ids'][0]:
            return 'No matching sessions found.'
            
        formatted = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]  # This contains title, excerpt, track, location info
            distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] else None
            
            # Calculate relevance score
            relevance = round((1 - distance) * 100, 1) if distance is not None else "N/A"
            
            # Build formatted session information with all available metadata
            session_info = f"""
Title: {metadata.get('title', 'Untitled')}
Date: {metadata.get('date_iso', 'No date')}"""

            # Add formatted date if available (may be more readable)
            if metadata.get('day_of_week'):
                session_info += f"\nDay: {metadata.get('day_of_week')}"
                
            # Add time if available
            if metadata.get('time'):
                session_info += f"\nTime: {metadata.get('time')}"
                
            # Add place/location if available
            if metadata.get('place'):
                session_info += f"\nLocation: {metadata.get('place')}"
                
            # Add track information
            session_info += f"\nTrack: {metadata.get('track', 'No track')}"
            
            # Add keywords if available
            if metadata.get('keywords'):
                session_info += f"\nKeywords: {metadata.get('keywords')}"
            
            # Add relevance score
            session_info += f"\nRelevance: {relevance}%"
            
            # Add the full description from metadata
            if metadata.get('description'):
                session_info += f"\nDescription: {metadata.get('description')}"
                
            # Add speaker information if available
            if metadata.get('speakers'):
                session_info += f"\nSpeakers: {metadata.get('speakers')}"
            
            
            # Add subsession information if available
            if metadata.get('subsessions') and metadata.get('subsessions') != '':
                try:
                    subsessions = json.loads(metadata.get('subsessions'))
                    if subsessions:
                        session_info += "\n\nSubsessions:"
                        for subsession in subsessions:
                            session_info += f"\n- {subsession.get('name', 'Unnamed subsession')}"
                            if subsession.get('speakers'):
                                session_info += f" (Speakers: {subsession.get('speakers')})"
                except:
                    # If JSON parsing fails, just skip subsessions
                    pass
            
            formatted.append(session_info)
            
        return '---'.join(formatted)

    async def pipe(self, body: dict, __user__: dict, __request__=None) -> Union[str, Generator, Iterator]:
        try:
            # Add diagnostic logging at the start
            logger.info("==== PIPE FUNCTION CALLED ====")
            
            # Check if streaming was requested in the API call
            streaming_enabled = body.get('stream', True)  # Default to True for backward compatibility
            logger.info(f"Request with streaming_enabled={streaming_enabled}")
            
            # Initialize User object once
            try:
                user_obj = User(**__user__)
                logger.info(f"Created User object successfully: {user_obj.id if hasattr(user_obj, 'id') else 'no id'}")
            except Exception as e:
                logger.error(f"Error creating User object: {str(e)}")
                user_obj = __user__  # Fallback to original dict if User object creation fails
            
            if not self._check_collection_exists():
                error_msg = f"SIOP sessions collection not found. Please ensure the collection '{self.valves.COLLECTION_NAME}' is created and populated."
                logger.error(error_msg)
                return {'choices': [{'message': {'content': error_msg}}]}

            # Extract and format messages
            messages = body.get('messages', [])
            if not messages:
                logger.warning("No messages found in request")
                return {'choices': [{'message': {'content': 'No messages found in request'}}]}
                
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    formatted_msg = {
                        'role': str(msg.get('role', 'user')),
                        'content': str(msg.get('content', ''))
                    }
                    formatted_messages.append(formatted_msg)
                else:
                    formatted_messages.append({
                        'role': 'user',
                        'content': str(msg)
                    })

            if not formatted_messages:
                logger.warning("No properly formatted messages found in request")
                return {'choices': [{'message': {'content': 'No properly formatted messages found in request'}}]}

            # Get the last user query
            original_query = formatted_messages[-1].get('content', '')
            query = original_query.strip().lower()
            if not query:
                logger.warning("No query found in last message")
                return {'choices': [{'message': {'content': 'No query found in last message'}}]}

            # Check for special requests
            if "Generate a concise, 3-5 word title" in original_query:
                logger.info("Handling special title generation request")
                return {"title": "🔍 SIOP Sessions Search"}
                
            if "Generate 1-3 broad tags" in original_query:
                logger.info("Handling special tags generation request")
                return {"tags": ["Conference", "Schedule", "SIOP"]}
                
            # Fast path for simple greetings - avoid full pipeline for basic interactions
            greeting_responses = {
                "hello": "Hello! I'm your SIOP 2025 conference assistant. I can help you discover sessions based on your interests, plan your schedule, and navigate the conference. What would you like to know about the conference, or would you like me to search for specific topics or sessions on a particular day?",
                "hi": "Hi there! I'm your SIOP 2025 conference assistant. How can I help you today? I can search for sessions on specific topics or help you plan your schedule for the conference days (April 2-5, 2025).",
                "hey": "Hey! I'm your SIOP 2025 conference assistant. I can help you find sessions or speakers that match your interests. What are you looking for today?",
                "help": "I'm your SIOP 2025 conference assistant. Here's how I can help:\n\n- Search for sessions by topic, speaker, or keyword\n- Find sessions on a specific date (April 2-5, 2025)\n- Get details about specific sessions\n- Suggest sessions based on your interests\n\nWhat would you like to know about?",
                "who are you": "I'm your AI assistant for the SIOP 2025 conference in Denver. I can help you explore the schedule, find relevant sessions based on your interests, and plan your conference experience. What would you like to know about?",
                "what can you do": "I can help you with the SIOP 2025 conference by:\n\n- Finding sessions on topics you're interested in\n- Providing information about specific sessions\n- Helping you navigate the conference schedule\n- Suggesting sessions based on your preferences\n\nWhat would you like to know about the conference?"
            }
            
            # Check if query is a simple greeting
            for greeting, response in greeting_responses.items():
                if query == greeting or query.startswith(f"{greeting} "):
                    logger.info(f"Using fast path response for greeting: '{query}'")
                    return {'choices': [{'message': {'content': response}}]}

            # Main LLM processing workflow
            try:
                # Extract date using LLM
                date = await self._extract_date_llm(original_query, formatted_messages, user_obj, __request__)
                logger.info(f"Extracted date filter: {date}")
                
                # Enhance query using LLM and search for sessions
                enhanced_query = await self._enhance_query_llm(original_query, date, formatted_messages, user_obj, __request__)
                search_results = self._search_sessions(enhanced_query, date)
                
                if not search_results or not search_results.get('ids') or not search_results['ids'][0]:
                    no_results_msg = f"I couldn't find any sessions matching '{original_query}'"
                    if date:
                        no_results_msg += f" on {date}"
                    no_results_msg += ". Please try a different search query or date."
                    
                    logger.info(f"No matching sessions found for query: '{original_query}'")
                    return {'choices': [{'message': {'content': no_results_msg}}]}
                
                logger.info(f"Found {len(search_results['ids'][0])} matching sessions")
                
                # Format results and generate response
                formatted_results = self._format_results(search_results)
                
                # Format the date context for the prompt
                date_context = ""
                if date:
                    day_mapping = {
                        '2025-04-02': 'Wednesday',
                        '2025-04-03': 'Thursday',
                        '2025-04-04': 'Friday',
                        '2025-04-05': 'Saturday'
                    }
                    day_of_week = day_mapping.get(date, '')
                    if day_of_week:
                        date_context = f"{day_of_week}, April {date[8:10]}, 2025 ({date})"
                    else:
                        date_context = date
                
                # Format chat history for context
                history_context = ""
                if formatted_messages and len(formatted_messages) > 1:
                    history_context = "Previous conversation:\n"
                    # Include up to 3 previous exchanges but skip the most recent query
                    for msg in formatted_messages[:-1][-6:]:  # Get up to 3 exchanges (6 messages)
                        if isinstance(msg, dict) and msg.get('content'):
                            role = "User" if msg.get('role', 'user').lower() == 'user' else "Assistant"
                            content = msg.get('content', '')
                            history_context += f"{role}: {content}\n"
                
                # Create the user prompt with all context
                user_prompt = RESPONSE_GENERATION_USER_PROMPT.format(
                    date_context=date_context if date else 'None - showing sessions across all conference days',
                    history_context=history_context,
                    results=formatted_results,
                    query=original_query
                )
                
                # Generate LLM response
                response = await self._llm_call(
                    system_prompt=RESPONSE_GENERATION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.3,  # Lower temperature for more reliable/factual responses
                    stream=streaming_enabled,
                    __user__=user_obj,
                    __request__=__request__
                )
                
                # Handle streaming response
                if streaming_enabled:
                    complete_content = ""
                    async for chunk in response.body_iterator:
                        content = self._process_chunk(chunk)
                        if content:
                            complete_content += content
                    return {'choices': [{'message': {'content': complete_content}}]}
                else:
                    # For non-streaming, the response is already formatted
                    return response
                
            except Exception as e:
                error_msg = f"Error in LLM processing: {str(e)}"
                logger.error(error_msg)
                return {'choices': [{'message': {'content': error_msg}}]}
        
        except Exception as e:
            error_msg = f"Error in pipe function: {str(e)}"
            logger.error(error_msg)
            
            # Log the stack trace
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            return {'choices': [{'message': {'content': error_msg}}]}
        finally:
            logger.info("==== PIPE FUNCTION COMPLETED ====") 