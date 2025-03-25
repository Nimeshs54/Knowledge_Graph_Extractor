import os
import json
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from rdflib import Graph, URIRef, Literal, Namespace
import owlrl
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Neo4j Aura connection
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Verify Neo4j Aura connection
try:
    with neo4j_driver.session() as session:
        result = session.run("RETURN 1")
        print("Neo4j Aura connected:", result.single()[0] == 1)
except Exception as e:
    print("Neo4j Aura connection failed:", str(e))

# Global variables
selected_model = None
rdf_graph = None
EX = Namespace("http://example.org/")

def extract_triples(text, llm):
    """Extract entities and relationships as RDF triples using LLM."""
    # Use a plain string prompt, no ChatPromptTemplate
    prompt = (
        f"Extract entities and relationships from the text as RDF triples (subject-predicate-object). "
        f"Return the result as a JSON string: [{{'subject': 's', 'predicate': 'p', 'object': 'o'}}, ...]. "
        f"Example: 'Einstein developed Relativity' -> '[{{'subject': 'Einstein', 'predicate': 'developed', 'object': 'Relativity'}}]'. "
        f"If no triples, return '[]'. Return ONLY the JSON string. Text: {text}"
    )
    print("Formatted Prompt:", prompt)  # Debug the exact prompt
    raw_response = None
    try:
        raw_response = llm.invoke(prompt)
        print("Raw LLM Response (extract_triples):", raw_response)
        if isinstance(raw_response, str):
            triples = json.loads(raw_response)
        elif hasattr(raw_response, 'content'):
            triples = json.loads(raw_response.content)
        else:
            triples = raw_response
        if not isinstance(triples, list):
            raise ValueError("LLM response must be a list of triples")
        return triples
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)} - Raw Response: {raw_response}")
        return []
    except Exception as e:
        print(f"Error extracting triples: {str(e)} - Raw Response: {raw_response}")
        return []

def store_triples_in_neo4j(triples):
    """Store RDF triples in Neo4j Aura."""
    with neo4j_driver.session() as session:
        for triple in triples:
            subject = triple["subject"].replace(" ", "_")
            predicate = triple["predicate"].replace(" ", "_")
            obj = triple["object"].replace(" ", "_")
            session.run(
                f"MERGE (s:Entity {{name: $subject}}) "
                f"MERGE (o:Entity {{name: $object}}) "
                f"MERGE (s)-[r:{predicate}]->(o)",
                subject=subject, object=obj
            )

def perform_owl_reasoning():
    """Apply OWL reasoning on the RDF graph."""
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(rdf_graph)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global selected_model, rdf_graph
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    selected_model = request.form.get('model')
    if not selected_model:
        return jsonify({"error": "No model selected"}), 400

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and split document
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        full_text = " ".join(doc.page_content for doc in documents)
        print("Full Text Length:", len(full_text))  # Debug total length
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)  # Adjust chunk size as needed
        chunks = text_splitter.split_text(full_text)
        print("Number of Chunks:", len(chunks))

        # Set up LLM
        if selected_model == "llama":
            llm = Ollama(model="llama3.2")
        elif selected_model == "deepseek":
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                return jsonify({"error": "GROQ_API_KEY not found in .env"}), 500
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-qwen-32b")
        else:
            return jsonify({"error": "Invalid model selected"}), 400

        # Extract triples from all chunks
        all_triples = []
        rdf_graph = Graph()
        for i, chunk in enumerate(chunks):
            print(f"Processing Chunk {i+1}/{len(chunks)} (Length: {len(chunk)})")
            triples = extract_triples(chunk, llm)
            all_triples.extend(triples)
            for triple in triples:
                rdf_graph.add((
                    URIRef(EX + triple["subject"].replace(" ", "_")),
                    URIRef(EX + triple["predicate"].replace(" ", "_")),
                    Literal(triple["object"]) if not triple["object"].startswith("http") else URIRef(triple["object"])
                ))
        
        print("Total Extracted Triples:", len(all_triples))
        print("All Triples:", all_triples)

        # Store in Neo4j Aura
        store_triples_in_neo4j(all_triples)

        # Perform OWL reasoning
        perform_owl_reasoning()

        return jsonify({"message": "File processed and knowledge graph constructed", "model": selected_model}), 200
    else:
        return jsonify({"error": "Only PDF files are supported"}), 400

@app.route('/query', methods=['POST'])
def query():
    global rdf_graph, selected_model
    if rdf_graph is None or selected_model is None:
        return jsonify({"error": "Please upload a document first"}), 400
    
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Retrieve all triples from RDF graph
    sparql_query = """
    SELECT ?subject ?predicate ?object
    WHERE { ?subject ?predicate ?object }
    """
    rdf_results = list(rdf_graph.query(sparql_query))
    rdf_context = " ".join(f"{row.subject} {row.predicate.split('/')[-1]} {row.object}" for row in rdf_results)

    # Use LLM for RAG
    if selected_model == "llama":
        llm = Ollama(model="llama3.2")
    else:
        llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="deepseek-r1-distill-qwen-32b")
    
    prompt = (
        f"Use ONLY the following context to answer the question. Do not use external knowledge. "
        f"If the context doesn’t provide an answer, say 'I don’t have enough information from the document to answer this.' "
        f"Context: {rdf_context}\nQuestion: {question}\nAnswer:"
    )
    response = llm.invoke(prompt)
    answer = response if isinstance(response, str) else getattr(response, 'content', str(response))
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)