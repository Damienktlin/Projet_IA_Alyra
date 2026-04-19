import tensorflow as tf
import numpy as np
IMG_SIZE = (224, 224)

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def predict(model, image):
    #model = load_model(model)
    pred_type, pred_couleur, pred_rarete, pred_encrable = model.predict(image)

    return {
        "type": {"label": np.argmax(pred_type), "proba": float(np.max(pred_type))},
        "couleur": {"label": np.argmax(pred_couleur), "proba": float(np.max(pred_couleur))},
        "rarete": {"label": np.argmax(pred_rarete), "proba": float(np.max(pred_rarete))},
        "encrable": {"label": int(pred_encrable[0][0] > 0.5), "proba": float(pred_encrable[0][0])}
    }

type_labels = ["Personnage", "Action", "Objet", "Lieu"]
couleur_labels = ["Ambre", "Améthyste", "Émeraude", "Rubis", "Saphir", "Acier"]
rarete_labels = ["Commune", "Unco", "Rare", "Super Rare", "Légendaire"]

def decode_predictions(preds):
    return {
        "type": {"label": type_labels[preds["type"]["label"]], "proba": f"{preds['type']['proba']*100:.2f}%"},
        "couleur": {"label": couleur_labels[preds["couleur"]["label"]], "proba": f"{preds['couleur']['proba']*100:.2f}%"},
        "rarete": {"label": rarete_labels[preds["rarete"]["label"]], "proba": f"{preds['rarete']['proba']*100:.2f}%"},
        "encrable": {"label": bool(preds["encrable"]["label"]), "proba": f"{preds['encrable']['proba']:.2f}"}
    }