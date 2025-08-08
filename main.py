import pandas as pd
import yaml
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
import string
import re

# Einmalig benötigte Downloads
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("german"))

# 📥 ed8.yml laden (Struktur: ed8 > ANGER: [wort1, wort2, ...])
with open("ed8.yml", "r", encoding="utf-8") as f:
    ed8_yaml = yaml.safe_load(f)
    ed8_dict = ed8_yaml.get("ed8", {})

# 🧠 Funktion zur Tokenisierung mit Negationskomposita
def tokenize_with_negation(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)  # URLs entfernen
    text = re.sub(r"\\d+", "", text)      # Zahlen entfernen
    text = text.translate(str.maketrans("", "", string.punctuation))  # Satzzeichen entfernen

    tokens = wordpunct_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and t.isalpha()]

    # Negationsbigrams erzeugen
    neg_starts = ["nicht", "nichts", "kein", "keine", "keinen"]
    bigrams = []
    skip_next = False

    for i in range(len(tokens) - 1):
        if skip_next:
            skip_next = False
            continue
        if tokens[i] in neg_starts:
            bigrams.append(f"{tokens[i]}_{tokens[i+1]}")
            skip_next = True
        else:
            bigrams.append(tokens[i])

    # Letztes Wort hinzufügen, falls nicht Teil eines Bigrams
    if not skip_next:
        bigrams.append(tokens[-1])

    return bigrams

# 🔍 Hauptfunktion zur Emotionserkennung
def get_ed8_emotions(text):
    tokens = tokenize_with_negation(text)
    total_terms = len(tokens)

    raw_counts = {}
    for emotion, wordlist in ed8_dict.items():
        # Falls Wörter als Liste mit Kommata gespeichert wurden, in einzelne Wörter splitten
        if isinstance(wordlist, str):
            wordlist = wordlist.split(",")
        wordlist = [w.strip().lower() for w in wordlist]

        count = sum(1 for t in tokens if t in wordlist)
        raw_counts[emotion.lower()] = count

    # Normierte Scores
    norm_counts = {
        f"{k}.norm": v / total_terms if total_terms > 0 else 0
        for k, v in raw_counts.items()
    }

    raw_counts["terms"] = total_terms
    raw_counts.update(norm_counts)
    return raw_counts

# 📄 CSV einlesen
df = pd.read_csv("speeches_final.csv")  # Achte darauf, dass die Datei im gleichen Ordner liegt

# 🧪 Emotionen berechnen
results = df["speechContent"].astype(str).apply(get_ed8_emotions)
results_df = pd.DataFrame(results.tolist())

# 🧷 Zusammenführen und speichern
df_final = pd.concat([df, results_df], axis=1)
df_final.to_csv("speeches_with_ed8.csv", index=False)

print("✅ Analyse abgeschlossen und in 'speeches_with_ed8.csv' gespeichert.")
# Fertig! Die Emotionserkennung ist nun implementiert und die Ergebnisse sind in der CSV-Datei gespeichert.