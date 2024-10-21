# Wiek i Niepełnosprawności Rozpoznawanie ludzi na pasach - Projekt AI

### Opis projektu

Projekt ma na celu stworzenie modelu sztucznej inteligencji zdolnego do rozpoznawania wieku oraz wykrywania potencjalnych niepełnosprawności z obrazów ludzi na pasach. Model będzie trenowany na zestawie danych zawierających zdjęcia osób oznaczone kategoriami wiekowymi oraz informacjami na temat ich niepełnosprawności (np. problemy z widzeniem, poruszaniem się itp.).

Przewiduje się zastosowanie głębokich sieci neuronowych, w szczególności sieci konwolucyjnych (CNN), które są powszechnie używane do analizy obrazów. Projekt jest stworzony w PyTorch.

### Cele projektu

1. **Klasyfikacja wieku** – Rozpoznanie wieku osoby na podstawie zdjęcia.
2. **Rozpoznanie niepełnosprawności** – Wykrywanie potencjalnych niepełnosprawności na podstawie cech twarzy lub widocznych atrybutów fizycznych.
3. **Klasyfikacja niepełnosprawności** - Klasyfikacja w zależności od wskaźnika procentowego wzrostu ilości potrzebnego czasu na przejście
4. **Klasyfikacja całej osoby** - W zależności od procentowej ilości potrzebnego czasu na przejście
5. **Klasyfikajcja grupy pieszych** - Przeprowadzenie tych operacji dla całej grupy na zdjęciu i wybraniu max z ilości czasu jako potrzebną ilość.


---

## Plan:
detekcja wieku: DeepFace
detekcja wózków: Open Images Dataset V7


## Wymagania

Do uruchomienia projektu potrzebujesz:

- **Python 3.8 - 3.12 (polecany: 3.9)**
- **PyTorch**
- **torchvision**
- **NumPy**
- **Pandas**
- **OpenCV** (do przetwarzania obrazów)
- **deepface**
- **Matplotlib** (do wizualizacji)
- **scikit-learn** (do ewaluacji modelu)

### Instalacja

1. **Skonfiguruj wirtualne środowisko** (opcjonalnie, ale zalecane):

   ```bash
   pip install deepface opencv-python pandas
   pip install tf-keras
   ```
tip: jeśli masz kilka wersji pythona na urządzeniu
musisz doprecyzować wersję na którą pobierasz te biblioteki
np. 

py -3.9 -m pip install ...
