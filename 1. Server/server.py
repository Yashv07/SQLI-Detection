import http.server
import socketserver
import os
from urllib.parse import urlparse, parse_qs
from gensim.models import Word2Vec
model = Word2Vec.load('../models/cbow_model.bin')
import pickle
with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
import joblib
mnb_model = joblib.load('../models/mnb_model.joblib')


PORT = 8000
web_dir = "."  # Directory where index.html is located

def get_doc_embedding(doc, model, vector_size):
    words = doc.split()
    embeddings = [model.wv[word] for word in words if word in model.wv]
    if not embeddings:
        return [0] * vector_size 
    return sum(embeddings) / len(embeddings)

def cbow(str):
    vec=1000
    cbow_output = get_doc_embedding(str, model, vec)
    print(type(cbow_output))
    if not isinstance(cbow_output, list):
        return cbow_output.reshape(1, -1)
    else:
        return [cbow_output]

def normalize(str):
    normalized =scaler.transform(str)
    return normalized

def prediction(str):
    res= mnb_model.predict(str)
    probabilities = mnb_model.predict_proba(str)
    return res,probabilities

class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        # Check if 'textfield' parameter exists in the query
        if 'input' in query_params:
            text_value = query_params['input'][0]

            cbow_res=cbow(text_value)
            normalized_res = normalize(cbow_res)
            traffic_result,prob = prediction(cbow_res)
            id="red"
            if traffic_result[0]==0:
                id="green"

            output_content = f'''
            <p id="Entered">You entered: `"{text_value}"`</p>
            <br>
            <p id="vect">Vectorized results: 
                <br>
                {cbow_res[0]}
            </p>
            <br>
            <p id="norm">
            Normalized results: 
                <br>
                {normalized_res[0]}
            </p>
            <br>
            <p id="{id}">
            Predicted results: 
                <br>
                {traffic_result}
                <br>
                Probability : {prob}
            </p>
            '''


            
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open(os.path.join(web_dir, "index.html"), "rb") as f:
                content = f.read().decode("utf-8")
                content_with_output = content.replace('<div id="output"></div>', f'<div id="output">{output_content}</div>')
                self.wfile.write(content_with_output.encode("utf-8"))
        else:
            # If 'textfield' parameter is not set, serve the regular index.html
            super().do_GET()

# Change directory to the location of your index.html file
os.chdir(web_dir)

# Start the server
with socketserver.TCPServer(("", PORT), MyRequestHandler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()
