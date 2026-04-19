from fastapi import FastAPI, UploadFile, File
import shutil
from .model import predict, decode_predictions, load_model
from .preprocess import preprocess_image
from fastapi.responses import HTMLResponse
import base64
from contextlib import asynccontextmanager



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model = load_model("model/best_model_finetuned.keras")
    yield
    
app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Lorca.ai</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f5f5f5;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
            }

            .container {
                width: 500px;
                margin-top: 40px;
            }

            .header {
                background-color: #fff000;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 20px;
            }

            .card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                text-align: center;
            }

            input[type="file"] {
                margin: 10px 0;
            }

            button {
                background-color: #333;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }

            button:hover {
                background-color: #555;
            }
        </style>
    </head>

    <body>
        <div class="container">
            
            <div class="header">
                Lorca.ai
            </div>

            <div class="card">
                <h3>Analyser une carte</h3>
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input name="file" type="file" required>
                    <br>
                    <button type="submit">Analyser</button>
                </form>
            </div>

        </div>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict_api(file: UploadFile = File(...)):
    contents = await file.read()
    
    # encoder image en base64 pour affichage
    img_base64 = base64.b64encode(contents).decode("utf-8")

    # sauver temporairement
    with open("temp.jpg", "wb") as f:
        f.write(contents)

    model = load_model("model/best_model_finetuned.keras")
    img = preprocess_image("temp.jpg")
    preds = predict(model, img)
    result = decode_predictions(preds)

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial;
                background: #f5f5f5;
                display: flex;
                justify-content: center;
            }}

            .container {{
                width: 500px;
                margin-top: 40px;
            }}

            .header {{
                background-color: #fff000;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 20px;
            }}

            .card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                text-align: center;
            }}

            img {{
                max-width: 50%;
                border-radius: 10px;
                margin-bottom: 15px;
            }}

            ul {{
                list-style: none;
                padding: 0;
                font-size: 18px;
            }}

            li {{
                margin: 8px 0;
            }}

            .highlight {{
                font-weight: bold;
                color: #333;
            }}
        </style>
    </head>

    <body>
        <div class="container">

            <div class="header">
                <a href="/" style="text-decoration:none">
                Lorca.ai
                </a>
            </div>

            <div class="card">
                <img src="data:image/jpeg;base64,{img_base64}" />
            </div>

            <div class="card">
                <h3>Résultats</h3>
            <ul>
                <li style="font-size: 18px;">Type : <b>{result['type']['label']} <br> avec une probabilité de {result['type']['proba']}</b></li>
                <li style="font-size: 18px;">Couleur : <b>{result['couleur']['label']} <br> avec une probabilité de {result['couleur']['proba']}</b></li>
                <li style="font-size: 18px;">Rareté : <b>{result['rarete']['label']} <br> avec une probabilité de {result['rarete']['proba']}</b></li>
                <li style="font-size: 18px;">Encrable : <b>{result['encrable']['label']} <br> avec une probabilité de {result['encrable']['proba']}</b></li>
            </ul>
            <a href="/" style="display:inline-block;margin-top:10px;">⬅ Retour</a>
            </div>
        </body>
    </html>
    """