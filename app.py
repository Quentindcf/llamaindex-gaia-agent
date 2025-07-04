from dotenv import load_dotenv
import os
import asyncio
# Load variables from .env file
load_dotenv()

# Now you can access them like this
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# Optional: fail loudly if missing
if not openai_key or not tavily_key:
    raise EnvironmentError("Missing OpenAI or Tavily API keys in .env")

import gradio as gr
import requests
import re
import inspect
import pandas as pd
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from tools import WikipediaToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.tools.arxiv.base import ArxivToolSpec
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.readers.whisper import WhisperReader
from youtube_transcript_api import YouTubeTranscriptApi


# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
#test commit 2
# --- Basic Agent Definition ---
class BasicAgent:
    def __init__(self):
        self.llm = OpenAI(model="gpt-4.1", temperature=0)  # Or "gpt-4.1"

        whisper_reader = WhisperReader(model="whisper-1")

        def transcribe_audio_tool(file_path: str) -> str:
            """
            Use this to transcribe audio files (mp3, wav, etc.)
            Args:
            file_path (str): local path of the file to transcribe
            """
            docs = whisper_reader.load_data(file_path)
            return docs[0].text if docs else "[No transcription found]"
        
        def transcribe_youtube(url: str) -> str:
            """Fetch and return transcript from a YouTube video."""
            try:
                video_id = url.split("v=")[-1].split("&")[0]
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                text = " ".join([entry["text"] for entry in transcript])
                return text
            except Exception as e:
                return f"[YouTube Transcript Error] {str(e)}"
        
        self.whisper_tool = FunctionTool.from_defaults(
            fn=transcribe_audio_tool,
            name="WhisperTranscription",
            description="""Use this to transcribe audio files (mp3, wav, etc.)
            Args:
            file_path (str): local path of the file to transcribe
            """
            )
        
        self.youtube_tool = FunctionTool.from_defaults(fn=transcribe_youtube, name="TranscribeYouTube")

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

                        NEVER answer the question yourself. Only delegate to one of the agents in your `can_handoff_to` list.
                        """,
                            llm=self.llm,
                            tools=[],  
                            # verbose=True,
                            can_handoff_to=["WikipediaAgent", "ArxivAgent", "WebSearchAgent", "AudioAgent"],
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
                            tools=[self.whisper_tool, self.youtube_tool],
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
                            agents=[self.controller_agent, self.wiki_agent, self.arxiv_agent, self.web_agent, self.audio_agent],
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
        #         print(repr(event))  # full structured representation
        #         print("=================\n")
        #     return handler
        
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
        
def extract_final_answer(output: str) -> str:
    for line in output.strip().splitlines():
        if "FINAL ANSWER:" in line:
            return line.split("FINAL ANSWER:")[-1].strip()
    return "[Answer not found] This was the output: "+output


def download_attachment(file_name: str, save_dir: str = "./downloads") -> str:
    os.makedirs(save_dir, exist_ok=True)

    file_id = os.path.splitext(file_name)[0]  # removes .mp3, .pdf, etc.
    url = f"{DEFAULT_API_URL}/files/{file_id}"
    save_path = os.path.join(save_dir, file_name)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    else:
        raise Exception(f"Failed to download {file_name}: {response.status_code}")


def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    questions_answered_correctly = [0,2,4,7,9, 13, 15,16,17,19]
    questions_for_deeper_wiki = [8,10,12]
    audio_reasoning =  []
    computer_vision = [1,3,6]
    arxiv_search = [14]
    maths = [5, 11,18]
    test= [6]
    for ii in test:
        item = questions_data[ii]
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name")
        local_path = ""
        if file_name:
            local_path = download_attachment(file_name)
            print(f"File saved to {local_path}")

        general_prompt = '''You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
                            This is the question: '''
        final_prompt = general_prompt + question_text
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            result = agent(final_prompt, file_path=local_path)
            output = result.response.content if hasattr(result, "response") else str(result)
            submitted_answer = extract_final_answer(output)
            
            print(submitted_answer)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")
 
    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)