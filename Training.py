import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image


class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Modello AI")

        # Variabili
        self.annotations_path = tk.StringVar()
        self.epochs = tk.IntVar(value=50)

        # Label e pulsanti
        tk.Label(root, text="Seleziona il file delle annotazioni:").pack(pady=5)
        tk.Entry(root, textvariable=self.annotations_path, width=50, state="readonly").pack()
        tk.Button(root, text="Sfoglia", command=self.seleziona_file).pack(pady=5)

        tk.Label(root, text="Numero di Epoche:").pack(pady=5)
        tk.Entry(root, textvariable=self.epochs, width=10).pack()

        tk.Button(root, text="Avvia Training", command=self.avvia_training, bg="green", fg="white").pack(pady=10)

    def seleziona_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy Files", "*.npy")])
        if file_path:
            self.annotations_path.set(file_path)

    def avvia_training(self):
        file_path = self.annotations_path.get()
        num_epochs = self.epochs.get()

        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Errore", "Seleziona un file valido")
            return

        messagebox.showinfo("Training", f"Avvio training per {num_epochs} epoche...")
        self.train_model(file_path, num_epochs)

    def train_model(self, annotations_path, epochs):
        """ Addestra il modello usando il file delle annotazioni """
        data = np.load(annotations_path, allow_pickle=True)

        X_list = []  # Lista per immagini
        y_list = []  # Lista per coordinate bbox normalizzate

        for item in data:
            image_array = np.array(item["image"])  # Converti in numpy array
            image = Image.fromarray(image_array)  # Converti in immagine PIL
            image = image.resize((300, 200))  # Ridimensiona a (Larghezza, Altezza)
            image = np.array(image) / 255.0  # Normalizza

            objects = item["objects"]
            for obj in objects:
                bbox = obj["bbox"]
                x_center = (bbox[0] + bbox[2]) / 2 / 300  # Normalizzato
                y_center = (bbox[1] + bbox[3]) / 2 / 200  # Normalizzato
                X_list.append(image)  
                y_list.append([x_center, y_center])

        # Converte le liste in array NumPy
        X = np.array(X_list).reshape(-1, 200, 300, 3)
        y = np.array(y_list)

        # Definizione del modello CNN
        model = keras.Sequential([
            layers.Conv2D(32, (3,3), activation="relu", input_shape=(200, 300, 3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation="relu"),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(2, activation="sigmoid")  # Output normalizzato (x, y)
        ])

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Addestramento
        model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)

        # Salvataggio del modello
        model.save("modello.h5")
        messagebox.showinfo("Training", "Modello salvato come 'modello.h5'")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()
