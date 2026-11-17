# Defekterkennung an Gussteilen – vier Methoden im Vergleich

Binäre Bildklassifikation für die Qualitätskontrolle: Ist ein gegossenes Pumpenlaufrad **defekt** (`def_front`) oder **in Ordnung** (`ok_front`)?
Datensatz: [Real-life Industrial Dataset of Casting Product (Kaggle)](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product).

Verglichen werden vier Ansätze, von klassischer Bildverarbeitung bis zur Anomalieerkennung, die nur gute Teile kennt, sowie zwei Ensembles daraus. Die Auswertung legt den Schwerpunkt auf **übersehene Defekte**, weil ein defektes Teil beim Kunden teurer ist als ein Fehlalarm.

## Die zwei Daten-Splits

Der Kaggle-Datensatz enthält die Bilder in zwei Versionen. Daraus ergeben sich zwei Arten, die Daten aufzuteilen:

| | `official` (Kaggle-Split) | `clean` (leakage-frei) |
|---|---|---|
| **Bilder** | 7348 Bilder, 300×300 px | 1300 Originalbilder, 512×512 px |
| **Herkunft** | vom Datensatz-Autor aus den 1300 Originalen **augmentiert** (gedreht, gespiegelt, Helligkeit verändert) | die unveränderten Originalaufnahmen |
| **Aufteilung** | vom Autor vorgegebene Ordner `train/` (6633) und `test/` (715); Val = 15 % aus `train` | von mir stratifiziert 70 / 15 / 15 in Train (910), Val (195) und Test (195), Seed 42 |
| **Problem** | Varianten **desselben Teils** können in Train **und** Test liegen | jedes Teil liegt in genau einem Teil der Aufteilung |

**Was ist Data Leakage?** Wenn ein Modell im Training eine gedrehte Version eines Teils sieht und im Test eine gespiegelte Version desselben Teils bewertet, prüft der Test nicht mehr, ob das Modell *neue* Teile beurteilen kann. Teilweise erkennt es nur bekannte Teile wieder, und die Ergebnisse wirken besser, als sie sind. Das erklärt einen Teil der oft berichteten ~99–100 % auf diesem Datensatz.

**Warum beide Splits?**
- `official` ist mit anderen Arbeiten auf Kaggle vergleichbar und hat mehr Testbilder (715). Die Ergebnisse sind aber wahrscheinlich etwas zu optimistisch.
- `clean` ist der ehrlichere Test, weil das Modell dort wirklich unbekannte Teile bewertet. Mit 195 Testbildern ist er allerdings klein: Ein einzelnes Bild entspricht 0,5 Prozentpunkten.

Auch auf `clean` erreichen die besten Modelle ~99 %. Das Problem ist also wirklich gut lösbar, die hohen Werte sind nicht nur ein Leakage-Effekt.

## Ergebnisse (Testdaten)

**Leakage-freier Split `clean`** (nur die 1300 Originalbilder, 195 Testbilder, davon 117 defekt):

| Methode | ROC-AUC | Accuracy | Recall Defekt | Fehlalarme | Übersehene Defekte* |
|---|---|---|---|---|---|
| HOG/LBP + Logistische Regression | 0,9934 | 95,4 % | 96,6 % | 6,4 % | 0 (bei 23 % Fehlalarmen) |
| HOG/LBP + SVM (RBF) | 0,9997 | **99,0 %** | **99,1 %** | 1,3 % | 1 |
| Kleines CNN (von Grund auf) | 0,9966 | 97,4 % | 95,7 % | 0 % | 5 |
| ResNet18 eingefroren + LogReg | 0,9950 | 96,9 % | 98,3 % | 5,1 % | 2 |
| ResNet18 Fine-Tuning | **0,9999** | 97,9 % | 96,6 % | 0 % | 4 |
| PatchCore-Anomalieerkennung | 0,9504 | 80,0 % | 67,5 % | 1,3 % | 1 (bei 65 % Fehlalarmen) |
| Ensemble: Soft Voting | 0,9999 | **99,5 %** | **99,1 %** | **0 %** | 1 |
| Ensemble: Stacking | 0,9991 | **99,5 %** | **99,1 %** | **0 %** | 1 |

**Offizieller Kaggle-Split `official`** (715 Testbilder, davon 453 defekt):

| Methode | ROC-AUC | Accuracy | Recall Defekt | Fehlalarme | Übersehene Defekte* |
|---|---|---|---|---|---|
| HOG/LBP + Logistische Regression | 0,9927 | 96,2 % | 97,6 % | 6,1 % | 7 |
| HOG/LBP + SVM (RBF) | 0,9999 | 99,3 % | 99,1 % | 0,4 % | 4 |
| Kleines CNN (von Grund auf) | 1,0000 | 95,2 % | 92,5 % | 0 % | 6 |
| ResNet18 eingefroren + LogReg | 0,9985 | 98,6 % | 98,2 % | 0,8 % | 8 |
| ResNet18 Fine-Tuning | **1,0000** | **99,7 %** | **99,6 %** | **0 %** | **2** |
| PatchCore-Anomalieerkennung | 0,9608 | 85,6 % | 81,0 % | 6,5 % | 4 (bei 44 % Fehlalarmen) |
| Ensemble: Soft Voting | 1,0000 | 99,6 % | 99,3 % | 0 % | 3 |
| Ensemble: Stacking | **1,0000** | **99,9 %** | **99,8 %** | **0 %** | **1** |

Accuracy, Recall und Fehlalarme gelten bei der Standardschwelle (0,5).
\* Bei der High-Recall-Schwelle: auf den Validierungsdaten so gewählt, dass ≥ 99 % der Defekte erkannt werden, dann unverändert auf den Test angewendet und nie strenger als die Standardschwelle.
Alle Zahlen: [results/summary.md](results/summary.md). Grafiken: [notebooks/02_results.ipynb](notebooks/02_results.ipynb).

## Erkenntnisse

1. **Der Datensatz ist wegen der kontrollierten Aufnahmen gut lösbar.** Kamera, Licht und Lage des Teils sind immer gleich. Schon eine SVM auf handgemachten Kantenmerkmalen (HOG) erreicht ~99 %.
2. **Das Fine-Tuning von ResNet18 ist das beste Einzelmodell auf `official`.** Es übersieht 2 von 453 Defekten und erzeugt keinen Fehlalarm. Auf dem kleinen `clean`-Split liegt es gleichauf mit der SVM. Bei 195 Testbildern entspricht ein einzelnes Bild 0,5 Prozentpunkten, die Unterschiede dort sind also nicht belastbar.
3. **Die Ensembles sind auf beiden Splits am besten.** Das Stacking übersieht auf `official` nur 1 von 453 Defekten, ohne Fehlalarm (99,9 % Accuracy). Auf `clean` übersehen beide Ensembles 1 von 117 Defekten, ebenfalls ohne Fehlalarm (99,5 %). Die Modelle machen unterschiedliche Fehler, die sich gegenseitig ausgleichen. Das Stacking gewichtet vor allem die beiden ResNet-Varianten und gibt der Anomalieerkennung ein leicht negatives Gewicht: Ihre Information steckt schon in den anderen Modellen.
4. **AUC und Accuracy messen Verschiedenes.** Das kleine CNN trennt die Klassen auf `official` perfekt (AUC = 1,0), übersieht bei Schwelle 0,5 aber 7,5 % der Defekte. Mit einer auf Validierungsdaten gewählten Schwelle sind es nur noch 1,3 %, ohne einen einzigen Fehlalarm. Die Wahl der Schwelle ist in der Qualitätskontrolle eine eigene Designentscheidung.
5. **Die Anomalieerkennung braucht keine Defektbilder** und lokalisiert Ausbrüche und Kerben in ihren Heatmaps gut. Sie ist aber deutlich ungenauer, weil schon die guten Teile in Reflexion und Helligkeit stark schwanken.
6. **Grad-CAM bei ResNet18 ist grob** (5×5-Raster bei 160 px Eingabe) und liegt nicht immer auf dem Defekt. Die Erklärungen sollten mit höherer Auflösung genauer geprüft werden.

## Methoden

| # | Methode | Kurzbeschreibung | Datei |
|---|---|---|---|
| 1 | HOG/LBP + LogReg/SVM | Handgemachte Kanten- (HOG) und Texturmerkmale (LBP) auf 128 px, danach logistische Regression bzw. SVM mit RBF-Kernel | [src/baseline.py](src/baseline.py) |
| 2 | Kleines CNN | 4 Conv-Blöcke (16→96 Kanäle, 210 Tsd. Parameter), 128 px Graustufen, AdamW + Cosine-LR, Augmentierung (Spiegeln, 90°-Drehung, Helligkeit) | [src/train_cnn.py](src/train_cnn.py) |
| 3a | ResNet18 eingefroren | ImageNet-Merkmale (512 Werte) mit logistischer Regression | [src/transfer.py](src/transfer.py) |
| 3b | ResNet18 Fine-Tuning | Nur `layer4` und ein neuer Kopf werden trainiert (160 px), dazu Grad-CAM | [src/transfer.py](src/transfer.py) |
| 4 | PatchCore-Stil | Patch-Merkmale aus ResNet18 `layer2`+`layer3`, Memory Bank aus ~30 Tsd. Patches **nur von guten Teilen**; Score = Abstand des auffälligsten Patches zum nächsten bekannten Patch | [src/anomaly.py](src/anomaly.py) |

| 5a | Ensemble: Soft Voting | Mittelwert der kalibrierten Wahrscheinlichkeiten von SVM, CNN und beiden ResNet-Varianten | [src/ensemble.py](src/ensemble.py) |
| 5b | Ensemble: Stacking | Logistische Regression auf den kalibrierten Scores **aller** sechs Modelle; lernt, wie stark jedem Modell zu trauen ist | [src/ensemble.py](src/ensemble.py) |

**Zu den Ensembles:** Sie werden aus den gespeicherten Val- und Test-Scores der anderen Modelle gebaut, ohne neues Training. Weil die Scores auf verschiedenen Skalen liegen (Wahrscheinlichkeit, SVM-Abstand, Anomalie-Distanz), wird jedes Modell zuerst per Platt-Scaling auf den Val-Daten in eine Wahrscheinlichkeit umgerechnet. Die Val-Scores der Ensembles sind out-of-fold (5-fach Kreuzvalidierung), damit die Schwellenwahl nicht auf bereits gesehenen Daten passiert. Einschränkung: CNN und ResNet haben ihre beste Epoche ebenfalls auf Val gewählt, die Val-Scores dieser Mitglieder sind also leicht optimistisch.

Gemeinsame Hilfsfunktionen (Splits, Bild-Cache, Metriken, Schwellenwahl): [src/common.py](src/common.py) und [src/torch_utils.py](src/torch_utils.py).
Die Modellauswahl (beste Epoche) und die Schwelle werden **nur auf den Validierungsdaten** bestimmt. Der Test wird erst ganz am Ende einmal ausgewertet.

## Ausführen

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Die Daten von Kaggle in den Projektordner entpacken: `casting_data/casting_data/{train,test}/...` und `casting_512x512/casting_512x512/...`.

```bash
python run_all.py --smoke     # Funktionstest mit wenigen Bildern (~3 min)
python run_all.py             # alle Methoden auf beiden Splits (~1,5 h auf einer 2-Kern-Laptop-CPU)
python src/transfer.py --split clean --size 224 --epochs 10   # einzelne Methode mit eigenen Parametern
python src/evaluate.py        # Vergleichstabelle neu erzeugen
```

Alle Parameter zeigt `--help` an. Ergebnisse (JSON mit Scores, Grafiken, Modellgewichte) landen in `results/`.

## Struktur

```
src/        Methoden und gemeinsame Hilfsfunktionen
notebooks/  01_eda.ipynb (Datenanalyse), 02_results.ipynb (ROC-Kurven, Konfusionsmatrizen, Heatmaps)
results/    Metriken (JSON/CSV/MD), Grafiken, Modellgewichte
run_all.py  Startet alle Methoden auf beiden Splits
```

## Grenzen und nächste Schritte

- Einzelner Split mit festem Seed: Wiederholte Splits bzw. k-Fold auf `clean` würden Mittelwert ± Streuung liefern.
- Wegen der CPU wurden Bildgrößen (128/160 px) und Epochenzahlen bewusst klein gehalten. Mit GPU wären 224–300 px und längeres Fine-Tuning möglich.
- Das Leakage im offiziellen Split ist plausibel, aber nicht gemessen. Eine Suche nach nahezu doppelten Bildern (ähnlichster Trainingsnachbar je Testbild) würde es quantifizieren.
- Robustheit gegen Licht, Unschärfe und Rauschen wurde nicht getestet und wäre für einen Einsatz in der Produktion wichtig.
