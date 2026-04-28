from langchain_google_genai import ChatGoogleGenerativeAI
from app.langsmith.load_prompt import load_prompt_from_langsmith

# ✅ Har template ka apna LangSmith prompt name
TEMPLATE_PROMPT_MAP = {
    "srs": "prompt_srs",
    "sprintreport": "prompt_sprint_report",
    "wbs": "prompt_wbs",
    "testcase": "prompt_testcase",
    
    # fallback ke liye default
}

DEFAULT_PROMPT = "doc_prompt_pdf_selected"  # purana default


def get_prompt_name(template_name: str) -> str:
    """
    Template name se matching LangSmith prompt name return karo.
    Case-insensitive match. Agar koi match na mile toh default use karo.
    """
    key = template_name.strip().lower()
    return TEMPLATE_PROMPT_MAP.get(key, DEFAULT_PROMPT)


def generate_documentation(
    cleaned_pm_data: str,
    pdf_headings: list,
    selected_headings: list,
    template_name: str = ""        # ✅ naya param
):
    prompt_name = get_prompt_name(template_name)
    print(f"📄 [doc_agent] Template: '{template_name}' → Prompt: '{prompt_name}'")

    prompt = load_prompt_from_langsmith(prompt_name)
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    chain = prompt | llm

    result = chain.invoke({
        "cleaned_pm_data": cleaned_pm_data,
        "pdf_headings": pdf_headings,
        "selected_headings": selected_headings
    })

    return result.content if hasattr(result, "content") else str(result)


def create_docs_node(state: dict) -> dict:
    pm_data = state.get("pm_data", {})
    pdf_headings = state.get("pdf_headings", [])
    selected_headings = state.get("selected_headings", [])
    template_name = state.get("template_name", "")   # ✅ state se lo

    print("\n📝 [create_docs_node] PM data received:")
    print(pm_data)
    print(f"🗂️ Template: {template_name}")

    if not pm_data:
        return {"generated_docs": "⚠️ PM data is empty. Please check the Trello fetch step."}

    cleaned_pm_data = str(pm_data)

    docs = generate_documentation(
        cleaned_pm_data,
        pdf_headings,
        selected_headings,
        template_name=template_name   # ✅ pass karo
    )
    return {"generated_docs": docs}