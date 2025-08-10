import io
from pathlib import Path
import os

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # important on Streamlit Cloud (no GUI)
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
# from ollama_llm import OllamaLLM   # ← keep if you want to switch back locally
from groq_llm import GroqLLM        # ← use Groq

# Optional: for non-proportional resizing when "Keep aspect ratio" is OFF
from PIL import Image

# -------------------- Page & LLM --------------------
st.set_page_config(page_title="AI CSV Analyst", layout="wide")
st.title("🤖 Ask Anything About Your CSV (with Charts!)")

# Use Groq (OpenAI-compatible). Pick any Groq model you like:
# "llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"
try:
    llm = GroqLLM(model="llama-3.1-8b-instant")  # key resolved inside GroqLLM
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# If you want to switch back to local Ollama, comment the line above and uncomment:
# llm = OllamaLLM(model="llama3", api_base="http://localhost:11434")

# -------------------- Session state --------------------
if "chart_png" not in st.session_state:
    st.session_state.chart_png = None
if "answer_text" not in st.session_state:
    st.session_state.answer_text = None

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV uploaded successfully!")
        st.dataframe(df.head())

        prompt = st.text_area(
            "💬 Ask a question or request a chart:",
            placeholder="e.g., Create a bar chart of counts in Column B",
            height=100,
        )

        if st.button("Generate Answer"):
            if prompt.strip():
                with st.spinner("🧠 Thinking..."):
                    try:
                        smart_df = SmartDataframe(
                            df,
                            config={
                                "llm": llm,
                                "enable_code_execution": True,  # allow plotting code
                            },
                        )

                        # A) Clean matplotlib state each run
                        plt.close('all')

                        # Ask the model
                        response = smart_df.chat(prompt)

                        # B) Gentle retry if PandasAI returned its failure string
                        if isinstance(response, str) and "All objects passed were None" in response:
                            retry_prompt = (
                                prompt
                                + "\n\nIf a chart or dataframe is not straightforward, "
                                  "answer concisely in plain text. Do NOT return None."
                            )
                            plt.close('all')  # also start retry clean
                            response = smart_df.chat(retry_prompt)

                        # Minimal guard for a None reply
                        if response is None:
                            st.info("The model returned no structured answer for this prompt. Try rephrasing.")
                            st.session_state.answer_text = None
                            st.session_state.chart_png = None
                            st.stop()

                        # Store textual answer or show DataFrame directly
                        if isinstance(response, pd.DataFrame):
                            st.session_state.answer_text = None
                            st.dataframe(response)
                        else:
                            st.session_state.answer_text = response

                        # Capture the latest matplotlib figure (if any) and persist as PNG
                        fig = None
                        fig_nums = plt.get_fignums()
                        if fig_nums:
                            fig = plt.figure(fig_nums[-1])  # last created fig

                        if fig and fig.get_axes():
                            buf = io.BytesIO()
                            # render at decent DPI so zoom looks crisp
                            fig.set_dpi(150)
                            fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
                            buf.seek(0)
                            st.session_state.chart_png = buf.read()
                            plt.close(fig)
                        else:
                            st.session_state.chart_png = None
                            st.info("ℹ️ No chart was generated for this prompt.")

                    except Exception as e:
                        st.error(f"❌ Error during chat or chart rendering: {e}")
            else:
                st.warning("⚠️ Please enter a prompt.")

        # -------------------- Show answer --------------------
        if st.session_state.answer_text:
            st.success("✅ Answer:")
            st.write(st.session_state.answer_text)

        # -------------------- Zoom controls ABOVE the image --------------------
        if st.session_state.chart_png:
            # Controls live right above the image
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                width_px = st.slider("🔍 Zoom width (px)", 300, 2000, 900, step=50, key="zoom_width")
            with c2:
                keep_aspect = st.checkbox("Keep aspect ratio", True, key="keep_aspect")
            with c3:
                if not keep_aspect:
                    height_px = st.slider("Height (px)", 200, 1500, 500, step=20, key="zoom_height")
                else:
                    height_px = None  # ignored

            # Render persisted PNG with chosen zoom
            if keep_aspect:
                st.image(st.session_state.chart_png, width=width_px)
            else:
                # Resize to (width_px, height_px) using Pillow
                try:
                    img = Image.open(io.BytesIO(st.session_state.chart_png))
                    img = img.resize((int(width_px), int(height_px)), Image.BICUBIC)
                    out = io.BytesIO()
                    img.save(out, format="PNG")
                    out.seek(0)
                    st.image(out, width=None)  # already resized to exact pixels
                except Exception:
                    # Fallback if Pillow not available or fails
                    st.image(st.session_state.chart_png, width=width_px)

    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
