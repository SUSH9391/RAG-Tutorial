import json
import os

file_path = '1-rag_evaluation.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    
    # 1. Judge LLM in correctness metrics
    if 'eval_instructions =' in source and 'judge_llm =' in source and 'ChatGoogleGenerativeAI' in source:
        new_source = '''from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
    temperature=0.1, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)
judge_llm = ChatHuggingFace(llm=llm)

eval_instructions = "You are an expert professor specialized in grading students' answers to questions."

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    user_content = f"""You are grading the following question:
    {inputs['question']}
    Here is the real answer:
    {reference_outputs['answer']}
    You are grading the following predicted answer:
    {outputs['response']}
    Respond with CORRECT or INCORRECT:
    Grade:
    """
    response = judge_llm.invoke([
        {"role": "system", "content": eval_instructions},
        {"role": "user", "content": user_content}
    ]).content.strip()
    return "CORRECT" in response'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        cell['source'][-1] = cell['source'][-1][:-1]
        
    # 2. my_app definition
    elif 'def my_app(question' in source and 'ChatGoogleGenerativeAI' in source:
        new_source = '''from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."

def my_app(question: str, model: str = "mistralai/Mistral-7B-Instruct-v0.3", instructions: str = default_instructions) -> str:
    endpoint = HuggingFaceEndpoint(
        repo_id=model, 
        temperature=0.1, 
        huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    )
    llm = ChatHuggingFace(llm=endpoint)
    return llm.invoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": question},
    ]).content'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        cell['source'][-1] = cell['source'][-1][:-1]

    # 3. ls_target with gemini-2.0-flash
    elif 'def ls_target(' in source and 'gemini-2.0-flash' in source:
        new_source = source.replace('gemini-2.0-flash', 'mistralai/Mistral-7B-Instruct-v0.3')
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        cell['source'][-1] = cell['source'][-1][:-1]

    # 4. RAG Embeddings
    elif 'GoogleGenerativeAIEmbeddings' in source and 'vectorstore = InMemoryVectorStore' in source:
        new_source = source.replace('from langchain_google_genai import GoogleGenerativeAIEmbeddings', 'from langchain_huggingface import HuggingFaceEmbeddings')
        new_source = new_source.replace('GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv(\'GOOGLE_API_KEY\'))', 'HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")')
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        cell['source'][-1] = cell['source'][-1][:-1]
        
    # 5. RAG LLM
    elif 'llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash"' in source and source.strip().endswith('llm'):
        new_source = '''from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
    temperature=0.1, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)
llm = ChatHuggingFace(llm=endpoint)
llm'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        cell['source'][-1] = cell['source'][-1][:-1]

    # 6. Evaluators CorrectnessGrade
    elif 'CorrectnessGrade' in source and 'grader_llm' in source and 'ChatGoogleGenerativeAI' in source:
        new_source = '''import json
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

grader_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
    temperature=0.1, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
))

correctness_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 
Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
You MUST output your answer in valid JSON format like this exactly:
{"explanation": "your reasoning here", "correct": true}
"""

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    answers = f"QUESTION: {inputs['question']}\\nGROUND TRUTH ANSWER: {reference_outputs['answer']}\\nSTUDENT ANSWER: {outputs['answer']}"
    response = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": answers}
    ]).content
    
    try:
        grade = json.loads(response.strip().replace("```json", "").replace("```", ""))
        return grade.get("correct", False)
    except:
        return False'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        cell['source'][-1] = cell['source'][-1][:-1]

    # 7. Evaluators RelevanceGrade
    elif 'RelevanceGrade' in source and 'relevance_llm' in source and 'ChatGoogleGenerativeAI' in source:
        new_source = '''import json
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

relevance_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
    temperature=0.1, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
))

relevance_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a STUDENT ANSWER. 
Ensure the STUDENT ANSWER is concise and relevant to the QUESTION.
You MUST output your answer in valid JSON format like this exactly:
{"explanation": "your reasoning here", "relevant": true}
"""

def relevance(inputs: dict, outputs: dict) -> bool:
    answer = f"QUESTION: {inputs['question']}\\nSTUDENT ANSWER: {outputs['answer']}"
    response = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": answer}
    ]).content
    
    try:
        grade = json.loads(response.strip().replace("```json", "").replace("```", ""))
        return grade.get("relevant", False)
    except:
        return False'''
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        cell['source'][-1] = cell['source'][-1][:-1]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
