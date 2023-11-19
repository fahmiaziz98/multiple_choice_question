import streamlit as st
import random
from pipeline import Pipeline

st.header("Generate Multiple Choice QA Generation")
st.markdown(
    "I built this project based on this [paper](https://www.sciencedirect.com/science/article/pii/S0957417422014014#s0015), "
    "where they created End-to-End generation of Multiple-Choice questions using Text-to-Text transfer Transformer models (T5).\n\n"
    "This research focuses on using Transformer-based language models to automate the generation of multiple-choice questions (MCQs), "
    "with the aim of assisting or assisting educators in the process of creating reading comprehension (RC) assessments. "
    "This is relevant and timely as teachers can invest less time doing routine work and share more time with their students, "
    "thus building an engaging experience for face-to-face classroom interaction. "
    "This study addresses the issue of creating multiple-choice questionnaires from 3 viewpoints: QG, QA, and distractor generation (DG). "
    "An end-to-end pipeline for generating multiple-choice questions is proposed, based on a pre-trained T5 language model."
)



st.sidebar.info(
    "Note: The number of questions generated depends on the length of the context. "
    "You may find that the number of QA pairs does not match the number you want."
)

with st.sidebar:
    if "num_qa" not in st.session_state:
        st.session_state.num_qa = 5

    def on_change():
        st.session_state.num_qa = num_qa

    num_qa = st.slider("Select Number of QA questions", min_value=1, max_value=10, value=1, step=1, on_change=on_change)

if 'context' not in st.session_state:
    st.session_state.context = ""
st_text_area = st.text_area('Context to generate the QA', value=st.session_state.context, height=500)

def generate_qa():
    st.session_state.context = st_text_area
    mcq_generator = Pipeline()
    generator = mcq_generator.generate_mcqs(st_text_area, num_qa)
    st.session_state.generator = generator

# generate qa button
st_generate_button = st.button('Generate', on_click=generate_qa)

# Display generated MCQs in Streamlit
if hasattr(st.session_state, 'generator') and len(st.session_state.generator) > 0:
    st.subheader("Generated MCQs")
    for i, question in enumerate(st.session_state.generator, start=1):
        correct_answer = [question.answerText]
        distractors_subset = question.distractors[:3]  # Assuming you want 3 distractors
        options = correct_answer + distractors_subset

        # Shuffle options
        random.shuffle(options)

        options_with_labels = [{'label': chr(ord('A') + j), 'text': option} for j, option in enumerate(options)]

        st.write(f'Number {i}: {question.questionText}')
        for option in options_with_labels:
            if option["text"] == correct_answer[0]:
                st.write(f'<span style="color:green;">{option["label"]}. {option["text"]}</span>', unsafe_allow_html=True)
            else:
                st.write(f'{option["label"]}. {option["text"]}')
        st.write('-------------------')

