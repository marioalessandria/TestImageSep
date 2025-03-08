import os
import tensorflow as tf
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog

def carica_modello(percorso_modello):
    """Carica un modello TensorFlow salvato."""
    return tf.keras.models.load_model(percorso_modello, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})



def carica_immagini(cartella):
    """Carica tutte le immagini dalla cartella selezionata."""
    immagini = []
    for file_name in os.listdir(cartella):
        if file_name.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
            img_path = os.path.join(cartella, file_name)
            image = Image.open(img_path).convert("RGB")
            image = image.resize((256, 256))  # Adatta la dimensione dell'immagine se necessario
            image_array = np.array(image) / 255.0  # Normalizza l'immagine
            immagini.append((file_name, image_array))
    return immagini

#def inferisci(cartella_immagini, modello):
#    """Esegue l'inferenza sulle immagini."""
#    immagini = carica_immagini(cartella_immagini)
#    if not immagini:
#        print("Nessuna immagine trovata nella cartella.")
#        return
    
#    for nome_file, image_array in immagini:
#       # input_image = np.expand_dims(image_array, axis=0).reshape(1, -1)  # Appiattisce l'immagine
#        input_image = np.expand_dims(image_array, axis=0)  # Aggiunge la dimensione batch
#        predictions = modello.predict(input_image)
#        print(f"Inferenza su {nome_file}: {predictions}")##

def inferisci(cartella_immagini, modello):
    """Esegue l'inferenza sulle immagini della cartella selezionata."""
    immagini = carica_immagini(cartella_immagini)
    if not immagini:
        print("Nessuna immagine trovata nella cartella.")
        return
    
    for nome_file, image_array in immagini:
        input_image = np.expand_dims(image_array, axis=0)  # Aggiunge la dimensione batch
        predictions = modello.predict(input_image)
        print(f"Inferenza su {nome_file}: {predictions}")


def seleziona_cartella():
    """Apre una finestra per selezionare la cartella delle immagini."""
    cartella_selezionata = filedialog.askdirectory()
    if cartella_selezionata:
        print(f"Cartella selezionata: {cartella_selezionata}")
        inferisci(cartella_selezionata, modello)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale di Tkinter

    percorso_modello = filedialog.askopenfilename(title="Seleziona il modello TensorFlow", filetypes=[("Modelli Keras", "*.h5")])
    if percorso_modello:
        modello = carica_modello(percorso_modello)
        seleziona_cartella()
        print("Input del modello:", modello.input_shape)

    else:
        print("Nessun modello selezionato.")
