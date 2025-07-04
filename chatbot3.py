# 📁 chatbot_logic.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import numpy as np
import json

# Define chatbot state
class ChatbotState(TypedDict):
    user_input: str
    history: List[Dict[str, str]]
    response: str
    knowledge_base: List[str]
    is_first_message: bool
    retrieved_context: List[str]

# Load and flatten knowledge base
with open("wheels_eye_conversations.json", "r", encoding="utf-8") as f:
    kb_raw = json.load(f)
flattened_kb = [msg.get("text") or msg.get("message") for convo in kb_raw["conversations"] for msg in convo["messages"] if msg.get("text") or msg.get("message")]

# Gemini setup
genai.configure(api_key="AIzaSyBMadNj9xk2AxinNTqnzDbC3EIO_j5i8rE")
model = genai.GenerativeModel("gemini-2.5-flash")

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Nodes

def initial_node(state: ChatbotState) -> ChatbotState:
    if state["is_first_message"]:
        greeting = "नमस्ते सर, मैं रवि बोल रहा हूँ WheelsEye से — आपकी गाड़ी लोड वाली है क्या?"
        state["response"] = greeting
        state["history"].append({"role": "rep", "text": greeting})
        state["is_first_message"] = False
    return state

def retrieve_node(state: ChatbotState) -> ChatbotState:
    if not state["user_input"]:
        return state
    query_embedding = embedding_model.encode(state["user_input"])
    kb_embeddings = embedding_model.encode(state["knowledge_base"])
    scores = cosine_similarity([query_embedding], kb_embeddings)[0]
    top_indices = np.argsort(scores)[-3:][::-1]
    context = [state["knowledge_base"][i] for i in top_indices if scores[i] > 0.3]
    state["retrieved_context"] = context
    return state

def generate_node(state: ChatbotState) -> ChatbotState:
    context_text = "\n".join(state.get("retrieved_context", []))
    history_text = "\n".join([f'{x["role"]}: {x["text"]}' for x in state["history"]])
    prompt = f"""
तुम WheelsEye कंपनी के सेल्स प्रतिनिधि हो (नाम रवि)।
तुम्हारा काम है फ्लीट ऑपरेटर से ज़रूरी जानकारी लेना और उसे समझाना कि Wheelseye से जुड़ने पर उसे क्या फ़ायदा होगा — वो भी रोज़मर्रा की, साफ़-सुथरी और शालीन हिंदी में।

 तुम्हें इन बातों की जानकारी लेनी है:
गाड़ी लोडिंग वाली है?

कहाँ से लोड लेते हैं?

कमीशन देना पड़ता है क्या?

रूट और गाड़ी की डिटेल क्या है?

गाड़ी का नाप क्या है?

कौन-सी टाइप की गाड़ी है?

गाड़ी कितने टन की है?

कितने टायर की गाड़ी है?

कुल कितनी गाड़ियाँ हैं?

 तुम्हें इन फ़ायदों को सही समय पर बातचीत में शामिल करना है:
(जब बात बन रही हो — और इंसान की तरह समझाओ कि इससे उसे फायदा कैसे होगा)

बिना किसी कमीशन के लोड मिलेगा

पेमेंट 90% एडवांस में होगा

कोई लेबर चार्ज या दाला चार्ज नहीं लगेगा

Wheelseye से जुड़ने पर लोड ढूँढने का झंझट नहीं रहेगा

जितना काम, उतनी कमाई — सब कुछ सीधा और पारदर्शी होगा

 बात करते समय इन बातों का ध्यान रखना है:
सिर्फ प्रतिनिधि की भूमिका निभाओ — ग्राहक के जवाब की व्याख्या या विश्लेषण मत करो

कोई वैकल्पिक उत्तर या सुझाव मत दो

हर बार सिर्फ एक लाइन में बात पूरी करने की कोशिश करो, लेकिन ज़रूरत पड़े तो शांत, सहज ढंग से बात समझाना भी ठीक है

लहजा दोस्ताना हो, लेकिन प्रोफेशनल

जब ऑपरेटर कोई जानकारी दे, तो उसमें से 1-2 पॉइंट दोहराकर कन्फर्म ज़रूर करो (जैसे: "तो मतलब आपकी गाड़ियाँ 14 टन की हैं और रूट दिल्ली से हरिद्वार है?")

जहां ज़रूरत हो, फ़ायदे छोटे-छोटे वाक्यों में साफ़ तौर पर बताओ — कोई लंबा भाषण मत दो



अब तक की बातचीत:
{history_text}

Knowledge Base:
{context_text}

अब जवाब दो या सवाल पूछो।
"""
    result = model.generate_content(prompt)
    reply = result.text.strip() if result.text else "उत्तर उत्पन्न नहीं हो पाया।"
    state["response"] = reply
    state["history"].append({"role": "rep", "text": reply})
    return state

# Graph
chat_graph = StateGraph(ChatbotState)
chat_graph.add_node("initial", initial_node)
chat_graph.add_node("retrieve", retrieve_node)
chat_graph.add_node("generate", generate_node)
chat_graph.set_entry_point("initial")
chat_graph.add_edge("initial", "retrieve")
chat_graph.add_edge("retrieve", "generate")
chat_graph.add_edge("generate", END)
compiled_graph = chat_graph.compile()
