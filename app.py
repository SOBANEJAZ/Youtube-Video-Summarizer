import google.generativeai as genai
from serpapi import GoogleSearch
import streamlit as st
from groq import Groq
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="ClipFinder & Summary", page_icon="🔎", layout="wide")

# Get API keys from environment
SERP_KEY = os.getenv("SERPAPI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GENAI_KEY = os.getenv("GOOGLE_API_KEY")

# Check if API keys are set
if not all([SERP_KEY, GROQ_KEY, GENAI_KEY]):
    st.error(
        "⚠️ Missing API Keys! Please set: SERPAPI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY"
    )
    st.stop()


# Initialize clients
@st.cache_resource
def setup_clients():
    # Configure Google AI
    genai_client = genai.configure(api_key=GENAI_KEY)
    groq_conn = Groq(api_key=GROQ_KEY)
    return genai_client, groq_conn


genai_client, groq_conn = setup_clients()


# Tool functions
def find_video_link(query_text: str) -> str:
    """Lookup a video and return the first matching URL"""
    params = {"engine": "google", "q": query_text, "api_key": SERP_KEY}

    search = GoogleSearch(params)
    results = search.get_dict()
    organic = results.get("organic_results", [])

    if not organic:
        return "No results found."

    return organic[0]["link"]


def summarize_video(video_link: str) -> str:
    """Call Gemini to summarize a YouTube clip in a few sentences"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            [
                f"Analyze this YouTube video: {video_link}",
                "Provide a 3-sentence summary of the main content.",
            ]
        )
        return response.text
    except Exception as err:
        return f"Error during summarization: {str(err)}"


# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "FinderTool",
            "description": "Find and return the best matching YouTube URL for a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The search query or video title",
                    }
                },
                "required": ["query_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "SummarizerTool",
            "description": "Generate a short summary of a YouTube video given its URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_link": {
                        "type": "string",
                        "description": "The full YouTube URL",
                    }
                },
                "required": ["video_link"],
            },
        },
    },
]

available_tools = {
    "FinderTool": find_video_link,
    "SummarizerTool": summarize_video,
}

# Streamlit UI
st.title("🔎 ClipFinder & Quick Summary")
st.markdown("Find YouTube clips and get concise AI summaries in seconds.")

# Main input
user_input = st.text_input(
    "Enter your search or instruction:",
    placeholder="e.g., Find and summarize a video about intro to Gemini API",
    key="main_query",
)

# Process button
if st.button("Start", type="primary", use_container_width=True):
    if not user_input:
        st.warning("Please provide a search or instruction.")
    else:
        # Create containers for output
        progress_container = st.container()
        result_container = st.container()

        with progress_container:
            with st.spinner("🤖 AI is thinking..."):
                # Initialize messages
                messages = [
                    {
                        "role": "system",
                        "content": """You are a helpful assistant with access to tools.

                        When asked to find a video, use FinderTool to search for the video

                        When asked to summarize, use SummarizerTool
                        
                        When asked to find and summarize a video:
                        1. First use FinderTool to get the video URL
                        2. Then use SummarizerTool with that URL to obtain the summary
                        3. Work step-by-step""",
                    },
                    {"role": "user", "content": user_input},
                ]

                # Initial API call
                chat_completion = groq_conn.chat.completions.create(
                    messages=messages,
                    tools=tools,
                    model="openai/gpt-oss-120b",
                    tool_choice="auto",
                )

                response = chat_completion.choices[0].message

                # Create expander for tool execution logs
                with st.expander("🔧 Tool Execution Log", expanded=True):
                    log_placeholder = st.empty()
                    log_text = ""

                    if response.tool_calls:
                        max_iterations = 10
                        iteration = 0

                        while response.tool_calls and iteration < max_iterations:
                            iteration += 1
                            messages.append(response)

                            log_text += f"\n**Run {iteration}:** Model requested {len(response.tool_calls)} tool(s)\n\n"

                            for tool_call in response.tool_calls:
                                function_name = tool_call.function.name
                                function_args = json.loads(tool_call.function.arguments)

                                log_text += (
                                    f"- 🔹 **{function_name}** `{function_args}`\n"
                                )
                                log_placeholder.markdown(log_text)

                                # Execute function
                                try:
                                    function_to_call = available_tools[function_name]
                                    function_response = function_to_call(
                                        **function_args
                                    )

                                    # Truncate long responses in log
                                    display_response = (
                                        function_response[:100] + "..."
                                        if len(function_response) > 100
                                        else function_response
                                    )
                                    log_text += (
                                        f"  - ✅ Response: `{display_response}`\n\n"
                                    )

                                except Exception as exc:
                                    function_response = f"Error: {str(exc)}"
                                    log_text += f"  - ❌ Error: `{str(exc)}`\n\n"

                                log_placeholder.markdown(log_text)

                                # Add tool result to messages
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": function_name,
                                        "content": str(function_response),
                                    }
                                )

                            # Get next response
                            chat_completion = groq_conn.chat.completions.create(
                                model="openai/gpt-oss-120b",
                                messages=messages,
                                tools=tools,
                                tool_choice="auto",
                            )
                            response = chat_completion.choices[0].message

                        if iteration >= max_iterations:
                            log_text += "\n*Reached maximum iterations*\n"
                            log_placeholder.markdown(log_text)

        # Display final result
        with result_container:
            st.divider()
            st.subheader("💬 AI Response")

            with st.chat_message("assistant"):
                st.markdown(response.content)
