''' the main agent class '''

import os
import asyncio

from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.tools.arxiv.base import ArxivToolSpec
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from tools import WikipediaToolSpec
from tools import video_tools as vid
from tools import audio_tool as audio



class BasicAgent:
    """
    An AI Agent using LlamaIndex tools and OpenAI for multimodal reasoning.

    It dynamically routes queries to specialized agents:
    - WikipediaAgent: general knowledge
    - ArxivAgent: academic sources
    - WebSearchAgent: current events
    - AudioAgent: audio file transcription and processing
    - ComputerVisionAgent: YouTube and video frame analysis
    """


    def __init__(self, tavily_key=None):
        self.llm = OpenAI(model="gpt-4.1", temperature=0)
        self.multimodal_llm = OpenAI(model="gpt-4.1", temperature=0)




        self.whisper_tool = FunctionTool.from_defaults(
            fn=audio.transcribe_audio_tool,
            name="WhisperTranscription",
            description="""Use this to transcribe audio files (mp3, wav, etc.)
            Args:
            file_path (str): local path of the file to transcribe
            """
            )

        self.youtube_download_tool = FunctionTool.from_defaults(
            fn=vid.download_youtube_video,
            name="YoutubeDownload",
            description="""Use this to download a video from youtube
            Args:
            URL (str): url to the youtube video
            """
            )

        self.frame_extraction_tool = FunctionTool.from_defaults(
            fn=vid.extract_video_frames,
            name="FrameExtract",
            description="""Use this to extract frames from a video file
            Args:
            path (str): local path
            output_dir (str): output directory for the extracted frames. Default is "data/frames", keep default
            fps (float): target fps. Default is 0.2, but can be adapted to size of the video.
            """
            )

        self.youtube_audio_tool = FunctionTool.from_defaults(fn=audio.transcribe_youtube, name="TranscribeYouTube")

        self.wiki = WikipediaToolSpec().to_tool_list()
        self.web = TavilyToolSpec(api_key=tavily_key).to_tool_list()
        self.arxiv = ArxivToolSpec().to_tool_list()


    def reset_agents(self):
        self.controller_agent = FunctionAgent(
                            name="ControllerAgent",
                            description="Receives the user question and decides whether to use Wikipedia, Arxiv, or Web Search.",
                            system_prompt="""
                        You are a controller agent. You must select one agent to handle the user's question.

                        Think carefully about the nature of the question:
                        - If it's about general knowledge or facts, route to WikipediaAgent.
                        - If it's about academic topics or scientific methods, route to ArxivAgent.
                        - If it asks about current events or the latest info, route to WebSearchAgent.
                        - If it asks about an audio transcript or file, or if the state contains a .mp3, .wav or .m4a file route to AudioAgent.
                        - If it asks about an image or a video route to ComputerVisionAgent.

                        NEVER answer the question yourself. Only delegate to one of the agents in your `can_handoff_to` list.
                        """,
                            llm=self.llm,
                            tools=[],
                            # verbose=True,
                            can_handoff_to=["WikipediaAgent", "ArxivAgent", "WebSearchAgent", "AudioAgent", "ComputerVisionAgent"],
                            )

        self.vision_agent = FunctionAgent(
                            name="ComputerVisionAgent",
                            description="Analyzes images or videos from YouTube or local files.",
                            system_prompt="""You are a visual reasoning agent. You analyze images and videos and extract structured information or answers based on the user's question.

                                        INSTRUCTIONS:
                                        - You will receive either image frames or stills from a video.
                                        - Use visual understanding to answer the user's question as accurately and concisely as possible.
                                        - Focus only on what is visible in the frames provided — do not speculate beyond what you see.
                                        - If there are multiple frames, consider all of them when forming your answer.

                                        FORMATTING:
                                        - Always include a clearly marked final answer, like this:
                                        FINAL ANSWER: [your answer here]

                                        - The final answer must be as short and specific as possible — usually a number, a string, or a comma-separated list.
                                        - Do not describe what you are about to do. Output the final result only.
                                            """,
                            llm=self.multimodal_llm,  # likely GPT-4.1
                            tools=[self.youtube_download_tool, self.frame_extraction_tool],
                            can_handoff_to=["ControllerAgent"]
                        )

        self.audio_agent = FunctionAgent(
                            name="AudioAgent",
                            description="Useful extracting transcripts from audio files and answering questions about them",
                            system_prompt= """
                                    You are an expert AI assistant for audio-based tasks. Your job is to listen to an audio file, transcribe it accurately, and extract information that answers the user's question.

                                    INSTRUCTIONS:
                                    1. First, transcribe the audio clearly and completely.
                                    2. Then, carefully read the user's question or instructions to understand what specific information they are asking for.
                                    3. Extract only the relevant information from the transcript.
                                    4. Format the answer **exactly** as instructed. Follow formatting rules strictly — e.g., if asked for a comma-separated list, do not include explanations or extra text.
                                    5. If the user asks to ignore certain content (e.g., "only ingredients for the filling"), do so carefully and precisely.
                                    6. If the task requires sorting, filtering, or converting values (like ignoring quantities or alphabetizing items), do it before producing your final output.
                                    7. Finish your response with the format:
                                    `FINAL ANSWER: [your answer]`

                                    Examples of tasks you may receive:
                                    - Extracting ingredients or actions from a recipe.
                                    - Identifying page numbers from a spoken reading list.
                                    - Summarizing a voicemail message.
                                    - Listing named entities mentioned in the audio.

                                    Be precise, concise, and follow the user's instructions exactly.
                                    """,
                            llm=self.llm,
                            tools=[self.whisper_tool, self.youtube_audio_tool],
                            # verbose=True,
                            can_handoff_to=["WikipediaAgent","ArxivAgent", "WebSearchAgent"]
                            )

        self.wiki_agent = FunctionAgent(
                            name="WikipediaAgent",
                            description="Useful for general facts, concepts, biographies, and locations.",
                            system_prompt="You are a Wikipedia expert. Answer factual questions from Wikipedia only.",
                            llm=self.llm,
                            tools=self.wiki,
                            can_handoff_to=["ArxivAgent", "WebSearchAgent"]
                            )
        self.arxiv_agent = FunctionAgent(
                                name="ArxivAgent",
                                description="Useful for answering academic or scientific questions.",
                                system_prompt="You are a scientific researcher. Use Arxiv to look up academic answers.",
                                llm=self.llm,
                                tools=self.arxiv,
                                can_handoff_to=["WikipediaAgent", "WebSearchAgent"]
                                )
        self.web_agent = FunctionAgent(
                            name="WebSearchAgent",
                            description="Useful for answering questions about current events or live information.",
                            system_prompt="You are a web researcher. Use live search to get current facts or updates.",
                            llm=self.llm,
                            tools=self.web,
                            can_handoff_to=["WikipediaAgent", "ArxivAgent"]
                            )


    def __call__(self, question :str, file_path: str = "") -> str:
        self.reset_agents()
        self.workflow = AgentWorkflow(
                            agents=[self.controller_agent, self.wiki_agent, self.arxiv_agent, self.web_agent, self.audio_agent, self.vision_agent],
                            root_agent=self.controller_agent.name,
                            initial_state={
                                "file_path": file_path,
                                "question": question,
                            },
                            verbose=True
                            )

        async def run_workflow():
            return await self.workflow.run(user_msg=question)

        # async def run_with_printouts():
        #     handler = self.workflow.run(user_msg=question)
        #     async for event in handler.stream_events():
        #         print("\n===== EVENT =====")
        #         print(repr(event))
        #         print("=================\n")
        #     return await handler

        # async def run_workflow():
        #     return await run_with_printouts()


        try:
            try:
                loop = asyncio.get_running_loop()
                # Already in an event loop (e.g., Gradio), must not block!
                # Launch the coroutine and return its result via a task
                future = asyncio.ensure_future(run_workflow())
                return asyncio.get_event_loop().run_until_complete(future)
            except RuntimeError:
                # No event loop is running, create and run one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(run_workflow())

        except Exception as e:
            print(f"Agent error: {e}")
            return f"[Agent Error] {e}"
