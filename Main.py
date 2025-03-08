import tkinter as tk
from tkinter import Frame, Button, filedialog
import Acquisizione_Hd  # Modulo per l'acquisizione delle immagini da hard disk
import subprocess
import LabelMeToTF
import Inferenza_Immagini  # Importa il modulo di inferenza
import tensorflow as tf
import Training
import subprocess

def seleziona_modello():
    """Apre una finestra di dialogo per selezionare un modello addestrato."""
    modello_path = filedialog.askopenfilename(title="Seleziona il modello addestrato",
                                              filetypes=[("Modelli TensorFlow", "*.h5")])
    return modello_path




#def avvia_training():
#        print ("AvviaTrainingPremuto")
def avvia_training():
    subprocess.Popen(["python", "Training.py"])

class MainApp:
    def __init__(self, root):
        """Inizializza l'interfaccia grafica principale."""
        self.root = root
        self.root.title("Mario SuperVision AI")
        self.root.geometry("800x600")
        # Cambia lo sfondo della finestra principale
        self.root.configure(bg="#696969")  # Puoi usare nomi di colori o codici esadecimali
        # Esempio: cambia lo sfondo di un widget Label
        label = tk.Label(self.root, text="Identificazione e coordinate oggetti", bg="lightblue")
        label.pack()
        
        # Frame per i pulsanti sulla destra
        self.frame_comandi = Frame(self.root, width=200, bg='#606965')
        self.frame_comandi.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Pulsante per caricare immagini da Hard Disk
        self.btn_carica_hd = Button(self.frame_comandi, text="Carica Immagini HD", command=self.carica_immagini_hd)
        self.btn_carica_hd.pack(pady=10, padx=10, fill=tk.X)

        # Pulsante per aprire LabelMe
        self.btn_open_labelme = Button(self.frame_comandi, text="Apri LabelMe", command=self.apri_labelme)
        self.btn_open_labelme.pack(pady=10, padx=10, fill=tk.X)

        # Pulsante per convertire annotazioni LabelMe
        self.btn_convert_labelme = Button(self.frame_comandi, text="Converti Annotazioni", command=self.converti_annotazioni_labelme)
        self.btn_convert_labelme.pack(pady=10, padx=10, fill=tk.X)

        # Pulsante per avviare il training
        self.btn_Avvia_Training = Button(self.frame_comandi, text="Avvia Training", command=avvia_training)
        self.btn_Avvia_Training.pack(pady=10, padx=10, fill=tk.X)

        # Pulsante inferenza
        self.btn_inferenza = Button(self.frame_comandi, text="Avvia Inferenza", command=self.avvia_inferenza_con_modello)
        self.btn_inferenza.pack(pady=10, padx=10, fill=tk.X)
        
        # Pulsante di uscita
        self.btn_exit = Button(self.frame_comandi, text="Esci", command=self.root.quit)
        self.btn_exit.pack(pady=100, padx=10, fill=tk.X)
        
    def carica_immagini_hd(self):
        """Richiama la funzione di acquisizione immagini da Hard Disk."""
        cartella = Acquisizione_Hd.seleziona_cartella()
        if cartella:
            immagini = Acquisizione_Hd.carica_immagini(cartella)
            Acquisizione_Hd.visualizza_immagini(immagini)

    def apri_labelme(self):
        """Apre l'applicazione LabelMe."""
        try:
            subprocess.run(["labelme"], check=True)
        except FileNotFoundError:
            print("Errore: LabelMe non trovato. Assicurati che sia installato e accessibile dal terminale.")

    def converti_annotazioni_labelme(self):
        """Apre una finestra per selezionare le cartelle di input e output e avvia la conversione."""
        input_dir = filedialog.askdirectory(title="Seleziona la cartella con i JSON di LabelMe")
        if not input_dir:
            return
        output_dir = filedialog.askdirectory(title="Seleziona la cartella di output")
        if not output_dir:
            return
        
        LabelMeToTF.convert_labelme_to_tf(input_dir, output_dir)
        print("Conversione completata!")

    

    def avvia_inferenza_con_modello(self):
        """Permette di selezionare un modello e avviare l'inferenza su una cartella di immagini."""
        modello_path = seleziona_modello()
        if not modello_path:
            print("Nessun modello selezionato.")
            return
        
        print(f"Modello selezionato: {modello_path}")
        model = tf.keras.models.load_model(modello_path)
        cartella_immagini = filedialog.askdirectory(title="Seleziona la cartella contenente le immagini")
        
        if not cartella_immagini:
            print("Nessuna cartella selezionata.")
            return
        
        print(f"Cartella immagini selezionata: {cartella_immagini}")
        Inferenza_Immagini.inferisci(cartella_immagini, model)
        print("Inferenza completata!")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
