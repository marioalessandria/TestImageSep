import cv2
import os
import tkinter as tk
from tkinter import filedialog

def seleziona_cartella():
    """Apre una finestra di dialogo per selezionare una cartella e restituisce il percorso."""
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale di Tkinter
    cartella = filedialog.askdirectory(title="Seleziona una cartella contenente immagini")
    return cartella

def carica_immagini(cartella):
    """Carica tutte le immagini dalla cartella specificata e le restituisce in una lista."""
    immagini = []
    estensioni_valide = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    if not os.path.exists(cartella):
        print("Errore: la cartella non esiste.")
        return []
    
    for file in sorted(os.listdir(cartella)):
        if any(file.lower().endswith(ext) for ext in estensioni_valide):
            percorso_completo = os.path.join(cartella, file)
            img = cv2.imread(percorso_completo)
            if img is not None:
                immagini.append((file, img))  # Salviamo anche il nome del file
            else:
                print(f"Impossibile caricare l'immagine: {file}")
    
    return immagini

def visualizza_immagini(immagini):
    """Mostra le immagini caricate una alla volta con una legenda sovraimpressa."""
    for i, (nome_file, img) in enumerate(immagini):
        img_copy = img.copy()
        titolo = f"Immagine {i+1}/{len(immagini)} - {nome_file}"
        legenda_spazio = "Premi SPAZIO per immagine successiva"
        legenda_q = "Premi Q per uscire"
        
        # Aggiunta della legenda sull'immagine
        cv2.putText(img_copy, legenda_spazio, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(img_copy, legenda_q, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(titolo, img_copy)
        print(f"{titolo}\nLegenda: Premi SPAZIO per l'immagine successiva, Q per uscire")
        tasto = cv2.waitKey(0) & 0xFF
        if tasto == ord('q'):
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cartella_selezionata = seleziona_cartella()
    if cartella_selezionata:
        immagini_caricate = carica_immagini(cartella_selezionata)
        print(f"{len(immagini_caricate)} immagini caricate con successo.")
        visualizza_immagini(immagini_caricate)
