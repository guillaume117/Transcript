from transformers import pipeline


def load_text_from_file(filename):
    """
    Charge le texte depuis un fichier texte.
    """
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def split_text(text, max_chars=2000):
    """
    Divise le texte en morceaux de taille maximale `max_chars` pour éviter
    des erreurs liées à la limite de tokens.
    """
    chunks = []
    while len(text) > max_chars:
        split_idx = text[:max_chars].rfind(". ") + 1  # Diviser au dernier point pour éviter de couper des phrases.
        if split_idx == 0:  # Pas de point trouvé dans la limite.
            split_idx = max_chars
        chunks.append(text[:split_idx].strip())
        text = text[split_idx:]
    chunks.append(text.strip())  # Ajouter le reste du texte.
    return chunks


def generate_summary_from_file(input_file, output_file="summary.txt"):
    """
    Génère un résumé à partir d'un fichier texte et sauvegarde le résumé.
    """
    # Charger le texte depuis le fichier
    text = load_text_from_file(input_file)
    
    # Diviser le texte en morceaux compatibles avec le modèle
    text_chunks = split_text(text)
    
    # Initialiser le pipeline de résumé
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)  # GPU spécifiée
    
    # Générer le résumé pour chaque morceau
    summaries = []
    for i, chunk in enumerate(text_chunks):
        try:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Erreur lors du résumé du segment {i + 1}: {e}")
    
    # Combiner tous les résumés
    final_summary = " ".join(summaries)
    
    # Sauvegarder le résumé dans un fichier texte
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(final_summary)
    
    print(f"Résumé sauvegardé dans {output_file}")


# Fichier d'entrée (transcription existante)
input_file = "transcription.txt"

# Générer le résumé
generate_summary_from_file(input_file)
