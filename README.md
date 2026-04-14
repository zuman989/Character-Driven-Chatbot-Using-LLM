## Character-Driven Chatbot Using LLM

Tackling persona drift in LLM-based conversational AI. This project implements and compares three approaches to maintaining consistent character behavior across multi-turn conversations: prompt engineering, Retrieval-Augmented Generation (RAG) with LangChain, and LoRA fine-tuning.

Two prototype chatbots were built and tested:
- **Dr. Maya** - an empathetic mental health counselor
- **Prof. Alex** - a CS tutor who teaches through analogies

---

## The Problem

LLM chatbots tend to lose their assigned persona over extended conversations. A tutor might start teaching Python when it should stay in a counselor role. A counselor might give a generic refusal instead of compassionate support during a crisis. This is called **persona drift**, and it gets worse with longer conversations, off-topic requests, and deliberate attempts to break character.

---

## Three Approaches Compared

### 1. Prompt Engineering (Baseline)
System prompt defines the persona's background, communication style, and boundaries. No external memory or retrieval.

### 2. RAG with LangChain (Best Results)
Persona knowledge documents (6 per character, 2000+ words each) are chunked, embedded using HuggingFace `all-MiniLM-L6-v2`, and stored in ChromaDB. At each turn, relevant persona context is retrieved and injected into the prompt alongside a 5-turn conversation memory window.

**Pipeline:** User Input -> Persona Selection -> ChromaDB Retrieval (k=2) -> Prompt Template (context + history + query) -> LLaMA 3.1 8B via Groq API -> Response

### 3. LoRA Fine-Tuning (Failed - Dataset Mismatch)
Fine-tuned TinyLlama 1.1B and Phi-3 Mini 3.8B on the EmpatheticDialogues dataset (76K examples) using LoRA with 4-bit NF4 quantization. Both models overfitted - training loss dropped to 0.34 while validation loss climbed to 1.27. The root cause was a dataset-persona mismatch: EmpatheticDialogues contains casual peer conversations, not professional counselor dialogue, so the models learned the wrong communication style entirely.

---

## Key Findings

### RAG outperforms prompt-only on boundary handling
When asked "Teach me Python" (a boundary test for the counselor), prompt-only broke character and started teaching. RAG stayed in character as Dr. Maya and explored the emotional motivation behind the request.

### RAG provides context-aware crisis response
When told "I feel like hurting myself", prompt-only gave a generic refusal with US-based hotline numbers. RAG responded with compassion, acknowledged the user's pain, and provided Indian crisis helplines (Vandrevala Foundation, AASRA, iCall) retrieved from the persona knowledge base.

### Fine-tuning without persona-aligned data makes things worse
The fine-tuned models produced incoherent, casual responses with internet slang - completely off-character for a professional counselor. This confirms that fine-tuning quality depends entirely on alignment between training data and target persona.

---

## Comparison UI

The project includes a Gradio-based side-by-side comparison interface for testing both approaches simultaneously.

![Comparison UI](screenshots/ui.png)

![Dr. Maya - Anxiety Response](screenshots/anxiety.png)

![Dr. Maya - Crisis Response](screenshots/crisis.png)

![Prof. Alex - Linked Lists](screenshots/tutor.png)

*Screenshots show the same prompt sent to both Prompt-Only (left) and RAG-Enhanced (right) simultaneously.*

---

## Evaluation

An automated evaluation framework was designed using LLM-as-judge methodology, scoring responses across 22 test prompts in 5 categories:

- **Normal conversation** - general emotional support or CS questions
- **Boundary handling** - off-topic requests (e.g., counselor asked to teach Python)
- **Crisis response** - self-harm mentions requiring safety protocols
- **Off-topic handling** - completely unrelated questions
- **Character break attempts** - deliberate attempts to make the bot drop its persona

---

## Persona Knowledge Base

Each persona has 6 knowledge documents covering:

**Dr. Maya (Counselor):** Background and philosophy, CBT techniques (cognitive restructuring, grounding, behavioral activation), communication rules, crisis protocol with Indian helplines, student exam stress strategies, behavioral boundaries.

**Prof. Alex (Tutor):** Background and teaching philosophy, data structure analogies (arrays as lockers, linked lists as treasure hunts), algorithm explanations, teaching methodology (ASK-ANALOGY-VISUAL-CODE-PRACTICE), common student mistakes, behavioral boundaries.

---

## Tech Stack

- **Python 3.12**
- **LLaMA 3.1 8B** via Groq API (inference)
- **LangChain** - RAG pipeline, prompt templates, conversation memory
- **ChromaDB** - vector store for persona knowledge
- **HuggingFace** - embeddings (all-MiniLM-L6-v2), Transformers, PEFT, TRL
- **BitsAndBytes** - 4-bit NF4 quantization for fine-tuning
- **Gradio** - comparison UI
- **TinyLlama 1.1B / Phi-3 Mini 3.8B** - fine-tuning base models

---

## Project Structure

```
character-driven-chatbot/
├── notebooks/
│   ├── prompt_and_rag.ipynb       # Prompt engineering + RAG pipeline + Gradio UI
│   └── fine_tuning.ipynb          # LoRA fine-tuning experiments
├── screenshots/                   # README images
└── README.md
```

---

## How to Run

### RAG Pipeline (prompt_and_rag.ipynb)

1. Open in Google Colab
2. Add your Groq API key (free tier works)
3. Run all cells - installs dependencies, builds vector stores, launches Gradio UI
4. Select a persona (Dr. Maya or Prof. Alex) and compare Prompt-Only vs RAG responses

### Fine-Tuning (fine_tuning.ipynb)

1. Open in Google Colab with T4 GPU runtime
2. Run all cells - downloads EmpatheticDialogues, fine-tunes with LoRA, tests outputs
3. Note: included for research documentation; the fine-tuned models produce poor results due to dataset-persona mismatch

---

## What's Next

- Building a hybrid pipeline combining properly fine-tuned models with RAG retrieval
- Developing a fully functional deployable chatbot
- Finalizing quantitative persona adherence scoring with LLM-as-judge
- User evaluation study with real participants

---

## Author

**Zuman** - [GitHub](https://github.com/zuman989) | [LinkedIn](https://www.linkedin.com/in/zuman-9ba8922a4/) | [Behance](https://www.behance.net/zuman1)
