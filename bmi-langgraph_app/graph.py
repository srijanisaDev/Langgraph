from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

class BMIState(TypedDict):
    weight: float
    height: float
    bmi: float
    category: str
    advice: str

def calculate_bmi(state: BMIState) -> BMIState:
    bmi = state['weight'] / (state['height'] ** 2)
    state['bmi'] = round(bmi, 2)
    return state

def classify_bmi(state: BMIState) -> BMIState:
    bmi = state['bmi']
    if bmi < 18.5:
        state['category'] = "Underweight"
    elif bmi < 25:
        state['category'] = "Normal"
    elif bmi < 30:
        state['category'] = "Overweight"
    else:
        state['category'] = "Obese"
    return state

def generate_advice(state: BMIState) -> BMIState:
    prompt = f"BMI is {state['bmi']} and category is {state['category']}. Give short and practical health advice in 2-3 sentences."
    state['advice'] = model.invoke(prompt).content
    return state

def build_graph():
    graph = StateGraph(BMIState)

    graph.add_node("calculate_bmi", calculate_bmi)
    graph.add_node("classify_bmi", classify_bmi)
    graph.add_node("generate_advice", generate_advice)

    graph.set_entry_point("calculate_bmi")
    graph.add_edge("calculate_bmi", "classify_bmi")
    graph.add_edge("classify_bmi", "generate_advice")
    graph.add_edge("generate_advice", END)

    return graph.compile()

bmi_graph = build_graph()
