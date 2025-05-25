"""
title: arXiv Research MCTS Pipe
description: Funtion Pipe made to create summary of searches uning arXiv.org for relevant papers on a topic and web scrape for more contextual information in a MCTS fashion.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
original MCTS implementation i based this project of: https://github.com/av // https://openwebui.com/f/everlier/mcts/
git: https://github.com/Haervwe/open-webui-tools  
version: 0.4.2
"""

import logging
import random
import math
import json
import aiohttp
import asyncio
from typing import List, Dict, Union, Optional, AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
from open_webui.constants import TASKS
from open_webui.models.users import User
from bs4 import BeautifulSoup
import re
from open_webui.main import generate_chat_completions


class TavilyApiError(Exception):
    pass


name = "Research Pipe"


def setup_logger():
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()


# Node class for MCTS
class Node:
    def __init__(self, **kwargs):
        self.id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=4))
        self.content = kwargs.get("content")
        self.parent = kwargs.get("parent")
        self.research = kwargs.get("research", [])
        self.exploration_weight = kwargs.get("exploration_weight", 1.414)
        self.max_children = kwargs.get("max_children", 3)
        self.children = []
        self.visits = 0
        self.value = 0
        self.score = 0
        self.temperature = kwargs.get("temperature", 1)
        self.depth = kwargs.get("depth", 1)

    def add_child(self, child: "Node"):
        child.parent = self
        self.children.append(child)
        return child

    def fully_expanded(self):
        return len(self.children) >= self.max_children

    def uct_value(self):
        epsilon = 1e-6
        if not self.parent:
            return float("inf")
        return self.value / (
            self.visits + epsilon
        ) + self.exploration_weight * math.sqrt(
            math.log(self.parent.visits) / (self.visits + epsilon)
        )

    def mermaid(self, offset=0, selected=None):
        padding = " " * offset

        # Sanitize content for Mermaid compatibility
        def sanitize_content(text):
            if not text:
                return "root"
            # Remove problematic characters and limit length
            sanitized = text[:25].replace("\n", " ")
            # Replace special characters that could break Mermaid syntax
            sanitized = re.sub(r'[(){}<>:"[\]]', "", sanitized)
            # Replace multiple spaces with single space
            sanitized = re.sub(r"\s+", " ", sanitized)
            # Ensure the text is not empty after sanitization
            return sanitized.strip() or "node"

        # Create node content
        content_preview = sanitize_content(self.content)

        # Create node ID and label
        node_label = f"{self.id}:{self.visits} - {content_preview}"
        # Escape any remaining special characters in the label
        node_label = node_label.replace('"', "&quot;")

        # Generate node definition
        msg = f'{padding}{self.id}["{node_label}"]\n'

        # Add styling if node is selected
        if selected == self.id:
            msg += f"{padding}style {self.id} stroke:#0ff,stroke-width:4px\n"

        # Generate children connections
        for child in self.children:
            msg += child.mermaid(offset + 4, selected)
            msg += f"{padding}{self.id} --> {child.id}\n"

        return msg


class MCTS:
    def __init__(self, **kwargs):
        self.topic = kwargs.get("topic")
        self.root = kwargs.get("root")
        self.pipe = kwargs.get("pipe")
        self.selected = None
        self.max_depth = kwargs.get("max_depth", 3)
        self.breadth = kwargs.get("breadth", 2)

    async def select(self):
        node = self.root
        while node.children:
            node = max(node.children, key=lambda child: child.uct_value())
        return node

    async def expand(self, node: Node, depth):
        await self.pipe.progress(f"Exploring research paths from {node.id}...")
        await self.pipe.emit_mermaid_diagram(self.mermaid(node))
        temperature = self.define_temperature(
            depth,
            node.score,
            self.max_depth,
            self.pipe.valves.TEMPERATURE_MAX,
            self.pipe.valves.TEMPERATURE_MIN,
            self.pipe.valves.DINAMYC_TEMPERATURE_DECAY,
        )
        for i in range(self.breadth):
            await self.pipe.emit_mermaid_diagram(self.mermaid())
            improvement = await self.pipe.get_improvement(node.content, self.topic)
            await self.pipe.emit_message(
                f"\nResearch direction {i+1}: {improvement}\n\n"
            )
            logger.debug(f"temperature:{temperature}")
            research = await self.pipe.gather_research(
                f"""Generate a new arXiv search query based on the improvement suggestion:
            Topic: {self.topic}
            Improvement: {improvement}"""
            )

            synthesis = await self.pipe.synthesize_research(
                research, self.topic, temperature
            )

            child = Node(
                content=synthesis,
                research=research,
                max_children=self.breadth,
                temperature=temperature,
            )
            node.add_child(child)

            await self.pipe.emit_mermaid_diagram(self.mermaid())

        return random.choice(node.children)

    def define_temperature(
        self,
        current_depth: int,
        parent_score: float,
        max_depth: int,
        temperature_max: float,
        temperature_min: float,
        dynamic: bool,
    ):
        if not self.pipe.valves.TEMPERATURE_DECAY:
            return 1

        if dynamic and parent_score > 0:
            # Inversely proportional to parent_score (higher temperature (creativity) for lower scores)
            score_normalized = parent_score / 10.0  # Normalize to 0-1 range
            scaling_factor = 1.0 + (1.0 - score_normalized) * (
                temperature_max - temperature_min
            )  # Scales with difference from ideal score
            temperature = (
                ((temperature_max - temperature_min) * (current_depth / max_depth))
                + temperature_min
            ) * scaling_factor
            # Clamp within bounds
            temperature_clamped = max(
                temperature_min, min(temperature, temperature_max)
            )

            return temperature_clamped

        else:  # Standard decay, not influenced by parent score
            temperature = temperature_max - (temperature_max - temperature_min) * (
                current_depth / max_depth
            )
            return temperature

    async def simulate(self, node):
        await self.pipe.progress(f"Evaluating research path {node.id}...")
        return await self.pipe.evaluate_content(node.content, self.topic)

    def backpropagate(self, node, score):
        while node:
            node.visits += 1
            node.value += score
            node.score = score
            node = node.parent

    def mermaid(self, selected=None):
        return f"""
```mermaid
graph LR
{self.root.mermaid(0, selected.id if selected else None)}
```
"""

    def best_child(self):
        return max(self.root.children, key=lambda child: child.visits)


EventEmitter = Callable[[dict], Awaitable[None]]


class Pipe:
    __current_event_emitter__: EventEmitter
    __current_node__: Node
    __question__: str
    __model__: str

    class Valves(BaseModel):
        MODEL: str = Field(
            default=None, description="Model to use (model id from ollama)"
        )
        TAVILY_API_KEY: str = Field(
            default="", description="API key for Tavily search service"
        )
        MAX_SEARCH_RESULTS: int = Field(
            default=3, description="Maximum number of search results to fetch per query"
        )
        ARXIV_MAX_RESULTS: int = Field(
            default=3, description="Maximum number of arXiv papers to fetch"
        )
        TREE_DEPTH: int = Field(
            default=4, description="Maximum depth of the research tree"
        )
        TREE_BREADTH: int = Field(
            default=3, description="Number of research paths to explore at each node"
        )
        EXPLORATION_WEIGHT: float = Field(
            default=1.414, description="Controls exploration vs exploitation"
        )
        TEMPERATURE_DECAY: bool = Field(
            default=True,
            description="Activates Temperature , lowers the Temperature in each subsequent step",
        )
        DINAMYC_TEMPERATURE_DECAY: bool = Field(
            default=True,
            description="Activates Temperature  Dynamic mapping, giving higher creativity for lower scored parent nodes",
        )
        TEMPERATURE_MAX: float = Field(
            default=1.4,
            description="Temperature for starting the MCTS process with Temperature decay ONLY if active",
        )
        TEMPERATURE_MIN: float = Field(
            default=0.5,
            description="Temperature the MCTS process will attempt to converge to with Temperature decay, if set to dinamic this value is not fixed",
        )
        TAVILY_MIN_SCORE_THRESHOLD: float = Field(
            default=0.70,
            description="Minimum relevance score (0.0-1.0) for Tavily search results to be included. Lower to include more results, raise for higher precision."
        )
        MCTS_MERMAID_UPDATE_MODE: str = Field(
            default='replace',
            description="How to update the MCTS Mermaid diagram in chat: 'replace' (update in place) or 'append' (post each version as a new message).",
            pattern="^(replace|append)$"
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:

        out = [{"id": f"{name}-{self.valves.MODEL}", "name": f"{name}"}]
        return out

    def resolve_model(self, body: dict) -> str:
        model_id = body.get("model")
        without_pipe = ".".join(model_id.split(".")[1:])
        return without_pipe.replace(f"{name}-", "")

    def resolve_question(self, body: dict) -> str:
        return body.get("messages")[-1].get("content").strip()

    async def search_arxiv(self, query: str) -> List[Dict]:
        """Gather research from arXiv"""
        if not query:
            logger.error("Empty query provided to arXiv search")
            return []

        await self.emit_status("tool", f"Fetching arXiv papers for: {query}...", False)
        try:
            arxiv_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"{query}",
                "max_results": self.valves.ARXIV_MAX_RESULTS,
                "sortBy": "relevance",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(arxiv_url, params=params) as response:
                    logger.debug(f"arXiv API response status: {response.status}")
                    if response.status != 200:
                        logger.error(f"arXiv API returned status {response.status}")
                        return []

                    data = await response.text()
                    if not data:
                        logger.error("Empty response from arXiv API")
                        return []

                    soup = BeautifulSoup(data, "xml")
                    entries = soup.find_all("entry")

                    if not entries:
                        logger.info("No entries found in arXiv response")
                        return []

                    results = []
                    for entry in entries:
                        try:
                            title = entry.find("title")
                            link = entry.find("link")
                            summary = entry.find("summary")

                            if not all([title, link, summary]):
                                logger.warning(
                                    "Incomplete entry found in arXiv response"
                                )
                                continue

                            results.append(
                                {
                                    "title": title.text,
                                    "url": link["href"],
                                    "content": summary.text,
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error processing arXiv entry: {e}")
                            continue

                    return results

        except aiohttp.ClientError as e:
            logger.error(f"arXiv API connection error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in arXiv search: {e}")
        return []

    async def search_web(self, query: str) -> List[Dict]:
        """Simplified web search using Tavily API"""
        if not self.valves.TAVILY_API_KEY:
            logger.warning("No Tavily API key provided")
            return []

        if not query:
            logger.error("Empty query provided to web search")
            return []

        # Check query length as per Tavily best practices
        if len(query) > 390: # 400 is the hard limit, 390 gives a small buffer
            logger.warning(f"Tavily query length ({len(query)} chars) exceeds recommended 390 chars. Consider shortening or breaking into sub-queries. Query: \"{query[:100]}...\"" )
            # Optionally, you could truncate here: query = query[:390]

        async with aiohttp.ClientSession() as session:
            try:
                url = "https://api.tavily.com/search"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.valves.TAVILY_API_KEY}"
                }
                data = {
                    "query": query,
                    "search_depth": "advanced",
                    "topic": "general",
                    "max_results": self.valves.MAX_SEARCH_RESULTS,
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_images": False,
                    "include_image_descriptions": False,
                    "include_domains": [],
                    "exclude_domains": [],
                }
                logger.debug(f"Tavily API request data: {data}")
                async with session.post(url, headers=headers, json=data) as response:
                    logger.debug(f"Tavily API response status: {response.status}")
                    if response.status != 200:
                        user_error_message = f"Tavily API error ({response.status}). Please check your Tavily API key configuration and ensure it is valid. If the issue persists, the Tavily service may be experiencing problems."
                        internal_error_message = f"Tavily API returned status {response.status} for query: {query}"
                        logger.error(internal_error_message)
                        raise TavilyApiError(user_error_message)

                    result = await response.json()
                    if not result:
                        user_error_message = "Tavily API returned an empty response. The service might be temporarily unavailable."
                        logger.error("Empty response from Tavily API")
                        raise TavilyApiError(user_error_message)

                    results = result.get("results", [])
                    if not results:
                        logger.info("No results found in Tavily response")
                        return []

                    raw_results_count = len(results)
                    logger.debug(f"Received {raw_results_count} results from Tavily before score filtering.")

                    processed_results = []
                    min_score_threshold = self.valves.TAVILY_MIN_SCORE_THRESHOLD

                    for result in results:
                        try:
                            current_score = result.get("score", 0.0)
                            if not all(
                                k in result
                                for k in ["title", "url", "content"] # Score is optional for logging but good to have
                            ) or current_score < min_score_threshold:
                                logger.warning(
                                    f"Incomplete result or score ({current_score}) below threshold ({min_score_threshold}) from Tavily API. Skipping result: {result.get('title', 'N/A')}"
                                )
                                continue
                            processed_results.append(
                                {
                                    "title": result["title"],
                                    "url": result["url"],
                                    "content": result["content"],
                                    "score": current_score,
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error processing Tavily result: {e}")
                            continue
                    
                    logger.debug(f"Returning {len(processed_results)} results from Tavily after score filtering (threshold: {min_score_threshold}).")
                    return processed_results

            except aiohttp.ClientError as e:
                user_error_message = "Could not connect to Tavily API. Please check your network connection."
                internal_error_message = f"Tavily API connection error: {e}"
                logger.error(internal_error_message)
                raise TavilyApiError(user_error_message)
            except json.JSONDecodeError as e:
                user_error_message = "Received an invalid response from Tavily API. The service might be temporarily unavailable."
                internal_error_message = f"Invalid JSON response from Tavily: {e}"
                logger.error(internal_error_message)
                raise TavilyApiError(user_error_message)
            except Exception as e:
                user_error_message = "An unexpected error occurred while searching the web."
                internal_error_message = f"Unexpected error in web search: {e}"
                logger.error(internal_error_message, exc_info=True)
                raise TavilyApiError(user_error_message)

    async def gather_research(self, topic: str) -> List[Dict]:
        """Gather initial research for the given topic"""
        await self.emit_status("tool", f"Researching...", False)

        # Preprocess the initial user query
        web_query, arxiv_query = await self.preprocess_query(topic)

        # Perform web search and arXiv search using the preprocessed queries
        web_research = await self.search_web(web_query)
        await self.emit_status(
            "tool", f"Web sources found:: {len(web_research)}", False
        )
        arxiv_research = await self.search_arxiv(arxiv_query)

        await self.emit_status(
            "tool", f"ArXiv papers found:: {len(arxiv_research)}", False
        )
        research = web_research + arxiv_research
        logger.debug(
            f"Research Result Created : ArXiv papers found: {len(arxiv_research)}, Web sources found: {len(web_research)}"
        )
        await self.emit_status(
            "user",
            f"Research gathered: ArXiv papers found: {len(arxiv_research)}, Web sources found: {len(web_research)}",
            True,
        )
        return research

    async def preprocess_query(self, query: str) -> tuple[str, str]:
        """Preprocess and enhance the initial user query"""
        if not query:
            logger.error("Empty query for preprocessing")
            return "", ""

        try:
            prompt_web = f"""
            Enhance the following query to improve the relevance of web search results:
            - Focus on adding relevant keywords, synonyms, or contextual phrases
            - The input query may be an initial vague request or an essay with proposed improvements
            - Only output the enhanced query, ready for an API call, without explanations or titles

            Initial query: "{query}"

            Enhanced web search query:
            """
            web_query = await self.get_completion(prompt_web)
            if not web_query:
                logger.error("Failed to generate web search query")
                return query, query

            prompt_arxiv = f"""
            Format an optimized query for the arXiv API based on the following input:
            - Use arXiv's query syntax (AND, OR, NOT) and search fields (ti, au, abs, cat)
            - Select appropriate categories from the provided list
            - The input may be an initial vague request or an essay with proposed improvements
            - Only output the formatted arXiv API query, without explanations or titles

            Initial query: "{query}"

            arXiv categories:
            - cs.AI: Artificial Intelligence
            - cs.LG: Machine Learning 
            - cs.CV: Computer Vision
            - cs.CL: Computation and Language (NLP)
            - cs.RO: Robotics
            - stat.ML: Machine Learning (Statistics)
            - math.OC: Optimization and Control
            - physics: Physics
            - q-bio: Quantitative Biology
            - q-fin: Quantitative Finance
            - econ: Economics

            Enhanced arXiv search query (API format):
            """
            arxiv_query = await self.get_completion(prompt_arxiv)
            if not arxiv_query:
                logger.error("Failed to generate arXiv query")
                return web_query or query, query

            return web_query, arxiv_query

        except Exception as e:
            logger.error(f"Query preprocessing failed: {e}")
            return query, query

    async def get_streaming_completion(
        self,
        messages,
        temperature: float = 1,
    ) -> AsyncGenerator[str, None]:
        try:
            form_data = {
                "model": self.__model__,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
            }

            logger.debug(f"Sending streaming request with model: {self.__model__}")
            logger.debug(f"Temperature: {temperature}")
            logger.debug(f"Message length: {len(str(messages))} chars")

            response = await generate_chat_completions(
                request=self.__request__,
                form_data=form_data,
                user=self.__user__
            )

            if not response:
                logger.error("No streaming response received from model")
                return

            if not hasattr(response, "body_iterator"):
                logger.error(f"Response type: {type(response)}")
                logger.error(f"Response content: {response}")
                return

            chunk_count = 0
            async for chunk in response.body_iterator:
                chunk_count += 1
                logger.debug(f"Processing chunk {chunk_count}")

                if not chunk:
                    logger.warning(f"Empty chunk received at position {chunk_count}")
                    continue

                try:
                    for part in self.get_chunk_content(chunk):
                        if part:
                            yield part
                        else:
                            logger.warning(f"Empty content in chunk {chunk_count}")
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_count}: {e}")

            logger.debug(f"Streaming completed. Processed {chunk_count} chunks")

        except Exception as e:
            logger.error(f"Streaming completion failed: {e}", exc_info=True)
            raise

    async def get_completion(self, messages) -> str:
        response = await generate_chat_completions(
            request=self.__request__,
            form_data={
                "model": self.__model__,
                "messages": [{"role": "user", "content": messages}],
            },
            user=self.__user__
        )

        if not response:
            logger.error("No response received from generate_chat_completions")
            return ""

        if "choices" not in response:
            logger.error(f"Unexpected response format: {response}")
            return ""

        return response["choices"][0]["message"]["content"]

    async def get_improvement(self, content: str, topic: str) -> str:
        """Get improvement suggestion"""
        prompt = f"""
    How can this research synthesis be improved?
    Topic: {topic}

    Current synthesis:
    {content}

    Suggest ONE specific improvement in a single sentence.
    """
        return await self.get_completion(prompt)

    async def synthesize_research(
        self, research: List[Dict], topic: str, temperature: float
    ) -> str:
        """Synthesize research content with streaming"""
        if not research:
            logger.warning("No research provided for synthesis")
            return "No research data available"

        if not topic:
            logger.error("No topic provided for synthesis")
            return "Topic required for synthesis"

        try:
            # Log input parameters
            logger.debug(f"Synthesizing research for topic: {topic}")
            logger.debug(f"Temperature: {temperature}")
            logger.debug(f"Number of research items: {len(research)}")

            research_text = ""
            try:
                research_items = []
                total_chars = 0
                for i, r in enumerate(research):
                    item_text = (
                        f"Source {i+1}:\n"
                        f"Title: {r.get('title', 'No title')}\n"
                        f"Content: {r.get('content', 'No content')}\n"
                        f"URL: {r.get('url', 'No URL')}\n"
                    )
                    total_chars += len(item_text)
                    research_items.append(item_text)
                    logger.debug(f"Source {i+1} length: {len(item_text)} chars")

                research_text = "\n\n".join(research_items)
                logger.debug(f"Total research text length: {total_chars} chars")

            except Exception as e:
                logger.error(f"Error formatting research text: {e}")
                return "Error: Failed to format research data"

            prompt = f"""
            Create a research synthesis on the topic: {topic}

            Available research:
            {research_text}

            Create a comprehensive synthesis that:
            1. Integrates the sources
            2. Highlights key findings
            3. Maintains academic rigor while being accessible
            """

            logger.debug(f"Final prompt length: {len(prompt)} chars")

            complete = ""
            chunk_count = 0
            try:
                async for chunk in self.get_streaming_completion(
                    [{"role": "user", "content": prompt}], temperature
                ):
                    chunk_count += 1
                    if not chunk:
                        logger.warning(f"Empty chunk {chunk_count} during synthesis")
                        continue
                    complete += chunk
                    await self.emit_message(chunk)

                logger.debug(f"Synthesis completed. Received {chunk_count} chunks")
                logger.debug(f"Final synthesis length: {len(complete)} chars")

                if not complete:
                    logger.error("No content generated during synthesis")
                    return "Error: No content generated"

            except Exception as e:
                logger.error(f"Streaming completion failed: {e}", exc_info=True)
                return f"Error generating synthesis: {str(e)}"

            return complete

        except Exception as e:
            logger.error(f"Research synthesis failed: {e}", exc_info=True)
            return f"Error in synthesis: {str(e)}"

    async def evaluate_content(self, content: str, topic: str) -> float:
        """Evaluate research content quality based on topic and content."""
        if not content or not topic:
            logger.error("Missing content or topic for evaluation")
            return 0.0

        logger.debug(f"Evaluating content for topic: {topic[:50]}...")

        try:
            prompt = f"""
            Evaluate the quality of the research synthesis provided below:

            Content: "{content}"
            Topic: "{topic}"

            Consider the following criteria:
            1. Integration of sources.
            2. Depth of analysis.
            3. Clarity and coherence.
            4. Relevance to the topic.

            Provide a single numeric score between 1 and 10, inclusive. 
            Do not include any explanation or additional text in your response—just the number.
            """

            result = await self.get_completion(prompt)
            if not result:
                logger.error("Empty evaluation result")
                return 0.0

            match = re.search(r"\b(10|\d(\.\d+)?)\b", result.strip())
            if not match:
                logger.error(f"No valid number found in evaluation response: {result}")
                return 0.0

            score = float(match.group())
            if not 1.0 <= score <= 10.0:
                logger.error(f"Score out of valid range: {score}")
                return 0.0

            return score

        except ValueError as e:
            logger.error(f"Error converting evaluation score: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during evaluation: {e}")
        return 0.0

    def get_chunk_content(self, chunk):
        if not chunk:
            logger.warning("Received empty chunk")
            return

        try:
            # Ensure chunk is a string
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode('utf-8')
            else:
                chunk_str = str(chunk) # Ensure it's a string if not bytes

            if chunk_str.startswith("data: "):
                chunk_str = chunk_str[6:]

            chunk_str = chunk_str.strip()
            logger.debug(f"Processing chunk: {chunk_str[:100]}...")

            if chunk_str == "[DONE]":
                logger.debug("Received [DONE] marker")
                return

            if not chunk_str:
                logger.warning("Empty chunk string after preprocessing")
                return

            try:
                chunk_data = json.loads(chunk_str)
                if not chunk_data:
                    logger.error("Empty chunk data after parsing")
                    return

                if "choices" not in chunk_data:
                    logger.error(f"No choices in chunk data: {chunk_data}")
                    return

                if not chunk_data["choices"]:
                    logger.error("Empty choices array in chunk data")
                    return

                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    content = delta["content"]
                    logger.debug(f"Extracted content: {content[:50]}...")
                    yield content
                else:
                    logger.debug(f"No content in delta: {delta}")

            except json.JSONDecodeError as e:
                logger.error(f'Chunk decode error: {e}. Chunk: "{chunk_str[:100]}"')

        except Exception as e:
            logger.error(f"Unexpected error processing chunk: {e}", exc_info=True)

    async def get_message_completion(self, model: str, content):
        async for chunk in self.get_streaming_completion(
            [{"role": "user", "content": content}]
        ):
            yield chunk

    async def stream_prompt_completion(self, prompt, **format_args):
        complete = ""
        async for chunk in self.get_message_completion(
            self.__model__,
            prompt.format(**format_args),
        ):
            complete += chunk
            await self.emit_message(chunk)
        return complete

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        try:
            if not body:
                logger.error("Empty request body")
                return "Error: Invalid request"

            model = self.valves.MODEL
            if not model:
                logger.error("No model specified in valves")
                return "Error: Model not configured"

            logger.debug(f"Model {model}")
            logger.debug(f"User: {__user__}")

            if "id" not in __user__:
                logger.error("User not authenticated")
                return "Error: User not authenticated"

            self.__user__ = User(**__user__)
            self.__request__ = __request__
            self.__current_event_emitter__ = __event_emitter__
            self.__model__ = model

            if __task__ and __task__ != TASKS.DEFAULT:
                logger.debug(f"Task: {__task__}")
                try:
                    response = await generate_chat_completions(
                        request=self.__request__,
                        form_data={
                            "model": model,
                            "messages": body.get("messages"),
                            "stream": False,
                        },
                        user=self.__user__
                    )
                    if not response or "choices" not in response:
                        logger.error("Invalid response from chat completion")
                        return "Error: Failed to generate response"
                    content = response["choices"][0]["message"]["content"]
                    return f"{name}: {content}"
                except Exception as e:
                    logger.error(f"Task completion failed: {e}")
                    return f"Error: {str(e)}"

            messages = body.get("messages", [])
            if not messages:
                logger.error("No messages in request")
                return "Error: No messages provided"

            topic = messages[-1].get("content", "").strip()
            if not topic:
                logger.error("Empty topic")
                return "Error: No topic provided"

            await self.progress("Initializing research process...")

            try:
                initial_temperature = (
                    self.valves.TEMPERATURE_MAX if self.valves.TEMPERATURE_DECAY else 1
                )

                initial_research = await self.gather_research(topic)
                if not initial_research:
                    logger.warning("No initial research results found")
                    return "Error: No research results found for the topic"

                logger.debug(f"Found {len(initial_research)} research items")

                # Add detailed logging before synthesis
                logger.debug(
                    f"Starting initial synthesis with temperature {initial_temperature}"
                )
                logger.debug(
                    f"Research items: {[r.get('title', 'No title') for r in initial_research]}"
                )

                initial_content = await self.synthesize_research(
                    initial_research, topic, initial_temperature
                )
                if not initial_content or initial_content.startswith("Error"):
                    logger.error(
                        f"Initial synthesis failed with content: {initial_content}"
                    )
                    return "Error: Initial research synthesis failed. Please try again with a different topic."

                # Log successful synthesis
                logger.debug(
                    f"Initial synthesis successful, content length: {len(initial_content)}"
                )

                root = Node(
                    content=initial_content,
                    research=initial_research,
                    max_children=self.valves.TREE_BREADTH,
                )

                mcts = MCTS(
                    root=root,
                    pipe=self,
                    topic=topic,
                    max_depth=self.valves.TREE_DEPTH,
                    breadth=self.valves.TREE_BREADTH,
                )

                best_content = initial_content
                best_score = -float("inf")

                for i in range(self.valves.TREE_DEPTH):
                    try:
                        await self.progress(
                            f"Research iteration {i+1}/{self.valves.TREE_DEPTH}..."
                        )

                        leaf = await mcts.select()
                        if not leaf:
                            logger.error("MCTS selection failed")
                            continue

                        child = await mcts.expand(leaf, i + 1)
                        if not child:
                            logger.error("MCTS expansion failed")
                            continue

                        score = await mcts.simulate(child)
                        mcts.backpropagate(child, score)

                        if score > best_score:
                            best_score = score
                            best_content = child.content

                    except Exception as e:
                        logger.error(f"Error in MCTS iteration {i}: {e}")
                        continue

                await self.emit_mermaid_diagram(mcts.mermaid())
                await self.emit_message(best_content)
                await self.done()
                return ""

            except Exception as e:
                logger.error(f"Research process failed: {e}", exc_info=True)
                return f"Error: Research process failed - {str(e)}"

        except Exception as e:
            logger.error(f"Pipe execution failed: {e}")
            return f"Error: {str(e)}"

    async def progress(self, message: str):
        await self.emit_status("info", message, False)

    async def done(self):
        await self.emit_status("info", "Research complete", True)

    async def emit_message(self, message: str):
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
        )

    async def emit_replace(self, message: str):
        await self.__current_event_emitter__(
            {"type": "replace", "data": {"content": message}}
        )

    async def emit_status(self, level: str, message: str, done: bool):
        if self.valves.MCTS_MERMAID_UPDATE_MODE == 'append':
            # In append mode, format status as a simple message string and use emit_message
            status_prefix = "[INFO]" # Default prefix
            if level == "tool":
                status_prefix = "[TOOL]"
            elif level == "user":
                status_prefix = "[USER_INFO]"
            elif level == "error": # Assuming you might have an error level
                status_prefix = "[ERROR]"
            
            formatted_message = f"{status_prefix} {message}"
            if done:
                formatted_message += " (Completed)"
            await self.emit_message(formatted_message) # Send as a regular chat message
        else:
            # Original behavior for replace mode
            await self.__current_event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )

    async def emit_mermaid_diagram(self, mermaid_content: str):
        if self.valves.MCTS_MERMAID_UPDATE_MODE == 'append':
            await self.emit_message(mermaid_content)
        else: # Default to replace
            await self.emit_replace(mermaid_content)
