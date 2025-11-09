from chromadb.api.fastapi import FastAPI
from ollama import ChatResponse, chat
from agentic_memory.memory_system import AgenticMemorySystem
from datetime import datetime
from flask import Flask, request, jsonify
import base64

memory_system = AgenticMemorySystem(
    model_name="all-MiniLM-L6-v2",  # Embedding model for ChromaDB
    llm_backend="ollama",  # LLM backend (openai/ollama)
    llm_model="gemma3:4b",  # LLM model name
)

possible_objects = "DOOR, BOTTLE, COMPUTER, RAG, SANDWICH, HEADPHONES"
important_properties = "open/closed"

def find_object_category(object, category):
    results = memory_system.search_agentic(object, k=5)
    for result in results:
        if (category in result['category']):
            return result['id']
    return ''

def init_object_schema(object, image = '') -> str:
    result = find_object_category(object, "schema")
    if result:
        return result

    response: ChatResponse = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": """List properties of a %s that would be useful to a person with Alzheimer’s, including the set of possible states for each property. This should follow JSON format and have this structure (example for Door). Also: (1) Add relevant properties visible in the attached image to the JSON schema. (2) Ensure that the following properties are in the schema: %s . Return JUST the JSON schema.
                    {
                        "object": {Door},
                        "states": {
                            "position": ["Open", "Closed"],
                            "type": ["Room", "Outdoor"],
                        }
                    }
                """
                % (object, important_properties),
                "images": [image],
            },
        ],
        format="json",
    )
    return memory_system.add_note(
        content=response.message.content,
        tags=["schema"],
        category="schema",
        timestamp=datetime.now(),  # YYYYMMDDHHmm format
    )

def get_object_data(object, schema, image):
    response: ChatResponse = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": """Using the following image of %s attached and the JSON schema, replace the options with the appropriate state for each JSON property. Return JUST the JSON object, surrounded by {}. %s
                    """
                           % (object, schema),
                "images": [image],
            },
        ],
        format="json",
    )
    return response.message.content

def update_object_info(object, content):
    result = find_object_category(object, "object")
    if result:
        memory_system.update(result, content=content)
        return result

    return memory_system.add_note(
        content=content,
        tags=["obj"],
        category="object",
        timestamp=datetime.now(),  # YYYYMMDDHHmm format
    )

def get_object_info(object):
    result_id = find_object_category(object, "object")
    return memory_system.read(result_id)

def get_object_from_query(query):
    response: ChatResponse = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": "Identify the object referenced in the following query: %s. Possible options: %s. ONLY output the identified object." % (query, possible_objects),
            },
        ],
    )
    return response.message.content

def result_from_query(query):
    results = memory_system.search_agentic(get_object_from_query(query), k=5)
    for result in results:
        if 'object' in result['category']:
            return result
    return ''

def search_from_query(query):
    result = result_from_query(query)
    if result == '':
        return "I cannot answer this."
    response: ChatResponse = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": "Answer the following query from the JSON data. Respond succinctly in one sentence. If you are not sure of the answer, do NOT hallucinate, respond 'I cannot answer this'. %s, %s" % (query, result['content'])
            },
        ],
    )
    return response.message.content


def test(object, image_address, query):
    with open(image_address, 'rb') as img_file:
        img_data = img_file.read()

    # Convert image to base64 for Ollama
    img_base64 = base64.b64encode(img_data).decode('utf-8')

    schema = init_object_schema(object, img_base64)
    print("SCHEMA:")
    print(schema) # which one?
    print(memory_system.read(schema).content)
    update_object_info(object, get_object_data(object, memory_system.read(schema).content, img_base64))

    obj = get_object_info(object)
    print("CONTENT:")
    print(obj.content)

    print("SEARCH: ")
    print(search_from_query(query))

def update_object(object, image):
    schema = init_object_schema(object, image)
    print("SCHEMA:")
    print(schema) # which one?
    print(memory_system.read(schema).content)
    update_object_info(object, get_object_data(object, memory_system.read(schema).content, image))

    obj = get_object_info(object)
    print("CONTENT:")
    print(obj.content)

def search_query(query):
    print("SEARCH: ")
    return search_from_query(query)


app = Flask(__name__)

@app.route('/object', methods=['POST'])
def object():
    # Parse the incoming JSON request data
    data = request.get_json()

    # Check if JSON data exists
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Extract parameters by their keys
    obj = data.get('object')
    img = data.get('image')

    # Validate that required keys exist
    if not obj or not img:
        return jsonify({"error": "Missing 'object' or 'image' key"}), 400

    # You now have the data
    # text_param = "the string sent by the client"
    # base64_image_data = "data:image/png;base64,iVBORw0KGgo..."

    # Let's run test(); TODO: Orchestrator

    update_object(obj, img)

    # Return a success response
    return jsonify({
        "message": "Data received :)",
        "text_length": len(obj),
        "image_data_length": len(img)
    }), 200

@app.route('/query', methods=['GET'])
def query():
    user_query = request.args.get('text')
    print("QUERYI: " + user_query)

    if user_query:
        answer = search_query(user_query)
        response_data = {
            "status": "success",
            "received_query": user_query,
            "answer": answer,
        }
        print(answer)
    else:
        return "Not available.", 400


if __name__ == '__main__':
    # test("headphones", "./IMG_0069.png", "are the headphones open?")
    app.run(port=9000)
