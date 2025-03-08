import os
import json
import numpy as np
from PIL import Image

def convert_labelme_to_tf(input_dir, output_dir):
    """
    Converte i file JSON di LabelMe in un formato compatibile con TensorFlow.
    
    :param input_dir: Cartella contenente i file JSON di LabelMe.
    :param output_dir: Cartella di output per i dati convertiti.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    annotations = []
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            json_path = os.path.join(input_dir, file_name)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            image_path = os.path.join(input_dir, os.path.basename(data["imagePath"]))
            
            if not os.path.exists(image_path):
                print(f"Errore: Immagine non trovata {image_path}")
                continue
            
            try:
                image = Image.open(image_path)
                image = image.convert("RGB")
                image_array = np.array(image)
            except Exception as e:
                print(f"Errore durante l'apertura dell'immagine {image_path}: {e}")
                continue
            
            objects = []
            for shape in data["shapes"]:
                if "label" not in shape or "points" not in shape:
                    print(f"Errore nel file {json_path}: dati mancanti")
                    continue

                label = shape["label"]
                points = shape["points"]
                x_min = min(p[0] for p in points)
                y_min = min(p[1] for p in points)
                x_max = max(p[0] for p in points)
                y_max = max(p[1] for p in points)
                
                objects.append({
                    "label": label,
                    "bbox": [x_min, y_min, x_max, y_max]
                })
            
            annotations.append({
                "image": image_array.tolist(),  # Converti l'array in lista per evitare problemi di serializzazione
                "objects": objects
            })
    
    output_file = os.path.join(output_dir, "annotations.npy")
    np.save(output_file, annotations, allow_pickle=True)
    print(f"Conversione completata. Dati salvati in {output_file}")
