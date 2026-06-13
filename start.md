# Research on Intelligent Recognition of Patient Behaviors Prior to Cognitive Impairment

## Research Thesis, Objectives, and Scope
**Hypothesis:** Deep learning algorithms based on skeletal data coordinates enable accurate and contactless prediction of movement anomalies and sudden patient falls while maintaining privacy.

**Objective:** To develop an intelligent system for automatic analysis of patient behavior based on silhouette and head detection, aimed at the early identification of clinical pathological patterns and the risk assessment of dangerous events.

**Scope:**
* Analysis of the state of the art in contactless pose estimation.
* Generation of a synthetic dataset of movement anomalies.
* Implementation of deep learning models for skeletal data extraction.
* Preparation of the application to the MUG Bioethics Committee (GUMed).
* Conducting accuracy testing and validation of algorithms.
* Analysis of the effectiveness of predicting the risk of dangerous events.

---

## Results
* Analysis of the state of the art in contactless human body pose estimation.
* Generation of a synthetic dataset of patient movement anomalies.
* Testing and validation of silhouette detection algorithms.
* Preparation of the application to the GUMed Bioethics Committee to obtain clinical data.

---

## Main Features
* A synthetic dataset generated using 3D environments and generative artificial intelligence models: Blender, Mixamo, and KlingAI.
* Behavioral analysis algorithms based on the MediaPipe library for pose estimation and classification models (Random Forest, LSTM network time windows).
* Analysis of video recorder and thermal imaging camera streams focused on keypoint extraction while strictly maintaining patient privacy.

---

## Future Works & Project Schedule
* Recording and analysis of real-world clinical data at the UCC department upon obtaining approval from the Bioethics Committee.
* Fine-tuning, optimization, and final validation of the effectiveness of AI algorithms in a hospital environment.

**Harmonogram (Plan Pracy):**

| Faza / Zadanie | Cel / Wynik | Termin |
| :--- | :--- | :--- |
| **Pilotażowe pozyskanie danych** | Rzeczywiste dane z monitoringu na oddziale hematologii UCK. Unikalny, zanonimizowany zbiór odzwierciedlający realne zachowania pacjentów. | X–XII |
| **Dane i adnotacje** | Rozbudowa datasetu o większą liczbę scen normalnych i anomalnych; ujednolicenie etykiet. Spójny zbiór gotowy do uczenia modeli. | X–XI |
| **Dane syntetyczne** | Dopracowanie scen 3D (różne kąty kamer, typy sylwetek, okluzje). Mniej jednorodny i bardziej realistyczny zestaw symulacji. | X–XII |
| **Modele** | Porównanie modeli bazowych: Random Forest, XGBoost oraz modeli sekwencyjnych. Wybór modelu najlepiej dopasowanego do celu. | XI–I |
| **Walidacja** | Testy na oddzielonych danych, analiza błędów oraz kontrola ryzyka przeuczenia. Wiarygodniejsza ocena skuteczności systemu. | XII–I |
| **Aspekty wdrożeniowe** | Opis procedury anonimizacji i założeń alertów. Uzupełnienie raportu końcowego o ograniczenia i rekomendacje. | N/A |

---

## 1. Rola Danych Syntetycznych
Dane syntetyczne rozumiane są w raporcie jako nagrania wygenerowane lub kontrolowane komputerowo, a nie bezpośrednio zebrane od rzeczywistych pacjentów. Takie podejście pozwala przygotować przykłady zdarzeń rzadkich, niebezpiecznych albo trudnych etycznie do rejestrowania w warunkach rzeczywistych. W kontekście diagnostyki pacjenta leżącego dotyczy to przede wszystkim anomalii ruchowych, które nie powinny być wywoływane wyłącznie po to, aby zebrać dane treningowe.

Syntetyczne nagrania umożliwiają kontrolę scenariusza ruchu: wiadomo, czy dana próbka przedstawia spokojne leżenie, kaszel, zwijanie się z bólu, upadek lub inny niepokojący ruch. Ułatwia to przypisanie etykiet oraz testowanie całego potoku przetwarzania danych. Jednocześnie takie dane nie zastępują w pełni danych rzeczywistych, ponieważ mogą być mniej zróżnicowane i mogą nie zawierać naturalnego szumu występującego w nagraniach z kamer monitorujących pacjenta.

**Tab. 1. Najważniejsze założenia przyjęte podczas przygotowania danych**

| Element | Założenie projektowe |
| :--- | :--- |
| **Obiekt obserwacji** | Postać człowieka wykonująca ruch związany ze zmianą położenia ciała. |
| **Jednostka analizy** | Krótkie nagranie wideo oraz sekwencja klatek opisanych punktami sylwetki. |
| **Klasa normalna** | Naturalna, bezpieczna aktywność ruchowa pacjenta (standardowe przemieszczanie się po sali, bezpieczne udawanie się do łazienki, wstawanie, siadanie, odpoczynek). |
| **Klasa anomalna** | Anomalie zwiastujące niebezpieczeństwo: nagłe upadki, utrata stabilności, osłabienie przy pionizacji, kliniczne wzorce przedupadkowe (opadanie na kolana, nagłe skurcze, utrata sił). |
| **Reprezentacja danych** | Zanonimizowane dane szkieletowe (współrzędne punktów kluczowych stawów i głowy). |

---

## 3. Zewnętrzny zbiór danych Kaggle
Jako jedno ze źródeł danych wykorzystano publiczny zbiór „Dataset Video For Human Action Recognition” udostępniony w serwisie Kaggle [1]. Zgodnie z opisem zbioru zawiera on nagrania wideo oraz dane szkieletowe związane z siedmioma klasami aktywności człowieka, m.in. siedzeniem, wstawaniem, chodzeniem, leżeniem oraz upadkiem.

Do dalszej pracy wybrano w szczególności sekwencje z klas „Fall down” (upadek), „Walking” (chodzenie) oraz „Standing up” (wstawanie). Najważniejsze było to, aby wybrane nagrania odzwierciedlały dynamikę przemieszczania się człowieka oraz momenty utraty stabilności postawy. Klasy związane z naturalną aktywnością (chodzenie, wstawanie) posłużyły do walidacji algorytmów w zakresie rozpoznawania zachowań bezpiecznych (**klasa normalna**). Z kolei sekwencje upadków były kluczowe dla procesu uczenia modeli w zakresie natychmiastowej detekcji nagłych anomalii ruchowych (**klasa anomalna**).

Zbiór Kaggle potraktowano jako uzupełnienie danych własnych. Jego główną zaletą jest gotowa struktura oraz obecność materiałów referencyjnych ułatwiających wstępne trenowanie modeli. Ograniczeniem jest natomiast fakt, że dataset zawiera ogólne, inscenizowane aktywności i nie był tworzony w warunkach klinicznych (wymagał selektywnego doboru i adaptacji).

**Tab. 2. Porównanie dwóch źródeł danych wykorzystanych w projekcie**

| Cecha źródła | Dataset Kaggle | Dane własne (Blender + Mixamo) |
| :--- | :--- | :--- |
| **Cel wykorzystania** | Baza referencyjna dla ogólnego ruchu (chodzenie, wstawanie, proste upadki). | Wygenerowanie scenariuszy anomalii, których brakuje w publicznym zbiorze. |
| **Kontrola ruchu** | Ograniczona (ruch wynika z istniejących nagrań). | Wysoka (możliwy wybór animacji i ustawienia sceny). |
| **Różnorodność** | Brak odzwierciedlenia specyfiki i motoryki ciężko chorego pacjenta. | Animacje mogą być zbyt idealne, co wymaga sztucznego zaburzania ich płynności. |
| **Zastosowanie** | Wstępne trenowanie modeli i ogólna walidacja potoku detekcji szkieletu. | Precyzyjne uczenie rozpoznawania konkretnych, groźnych sytuacji przedupadkowych. |

---

## 4. Własne dane generowane w Blenderze
Drugim i najważniejszym źródłem były dane syntetyczne wygenerowane samodzielnie w programie Blender. Do animacji wykorzystano i odpowiednio zmodyfikowano materiały z biblioteki Mixamo, obejmujące zarówno ruchy naturalne (wstawanie z łóżka, przemieszczanie się), jak i specyficzne anomalie behawioralne: nagłe upadki, utratę stabilności postawy.

Własne generowanie materiału pozwoliło na pełną kontrolę nad warunkami obserwacji. Możliwe było dowolne konfigurowanie kątów i wysokości wirtualnych kamer (symulujących rzeczywisty monitoring szpitalny), modyfikowanie typów sylwetek postaci oraz precyzyjne dawkowanie dynamiki ruchu.

> *Rys. 1. Klatka z nagrania wygenerowanego w środowisku Blender.*

---

## 5. Detekcja punktów sylwetki z nagrań wideo
Po przygotowaniu filmów wykonano detekcję punktów sylwetki. W notebooku `Wykrywanie_pozycji.ipynb` zastosowano bibliotekę MediaPipe Pose Landmarker oraz OpenCV. Kod działa na pojedynczym pliku wideo: wczytuje film, przetwarza kolejne klatki, wykrywa punkty charakterystyczne ciała i rysuje je na obrazie.

MediaPipe Pose Landmarker służy do wykrywania punktów ciała w obrazie lub filmie i może zwracać współrzędne punktów sylwetki w kolejnych klatkach [2]. W projekcie użyto trybu pracy dla wideo, dzięki czemu dla każdej klatki wyznaczany był znacznik czasu oraz zestaw punktów opisujących położenie wybranych części ciała.

> *Rys. 2. Przykład detekcji punktów sylwetki na syntetycznym nagraniu postaci, która upadła.*

**Fragment kodu odpowiadający za wykrywanie punktów kluczowych sylwetki:**

```python
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

result = landmarker.detect_for_video(mp_image, timestamp_ms)

if result.pose_landmarks:
    landmarks = result.pose_landmarks[0]
    for i, lm in enumerate(landmarks):
        all_landmarks.append({
            "frame": frame_idx,
            "landmark_id": i,
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility
        })
