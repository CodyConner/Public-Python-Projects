# pip install wordcloud matplotlib

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# --- Your terms (phrases kept intact) ---
terms = [
    "Forecasting",
    "Classification",
    "Big Data & Unstructured Data",
    "Image Processing",
    "Computer Vision",
    "Sentiment Analysis",
    "Clustering",
    "Factor Analysis",
]

# Option A: equal weights (all words same size)
frequencies = {t: 1 for t in terms}

# Option B (optional): tweak weights by importance
# frequencies = {
#     "Computer Vision": 5,
#     "Big Data & Unstructured Data": 4,
#     "Image Processing": 4,
#     "Classification": 3,
#     "Forecasting": 3,
#     "Sentiment Analysis": 3,
#     "Clustering": 2,
#     "Factor Analysis": 2,
# }

wc = WordCloud(
    width=1200,
    height=600,
    background_color=None,  # <-- transparent background
    mode="RGBA",            # <-- enables alpha channel
    prefer_horizontal=0.95,
    collocations=False,     # keep multi-word phrases intact
    margin=4,
).generate_from_frequencies(frequencies)

wc = wc.generate_from_frequencies(frequencies)

# Display and save (preserve transparency)
plt.figure(figsize=(12, 6), facecolor="none")
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("Wordcloud Traditional AI Terms.png", dpi=200, transparent=True)  # transparent PNG
plt.show()

input("Saved to Wordcloud Traditional AI Terms.png\n\nPress Enter to continue...")

terms = [
    "Natural Language Processing (NLP)",
    "Reinforcement Learning",
    "Neural Networks",
    "Large Language Models (LLMs)",
    "RAG (Retrieval Augmented Generation)",
    "Fine Tuning",
    "Embeddings",
    "Vector Database",
]

# Option A: equal weights (all words same size)
# frequencies = {t: 1 for t in terms}

# Option B (optional): tweak weights by importance
frequencies = {
    "Natural Language Processing (NLP)": 7,
    "Reinforcement Learning": 3,
    "Neural Networks": 8,
    "Large Language Models (LLMs)": 5,
    "RAG (Retrieval Augmented Generation)": 6,
    "Fine Tuning": 4,
    "Embeddings": 2,
    "Vector Database": 2,
}

wc = WordCloud(
    width=1200,
    height=600,
    background_color=None,  # <-- transparent background
    mode="RGBA",            # <-- enables alpha channel
    prefer_horizontal=0.95,
    collocations=False,     # keep multi-word phrases intact
    margin=4,
).generate_from_frequencies(frequencies)

wc = wc.generate_from_frequencies(frequencies)

# Display and save (preserve transparency)
plt.figure(figsize=(12, 6), facecolor="none")
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("Wordcloud New & Emerging Terms.png", dpi=200, transparent=True)  # transparent PNG
plt.show()

print("Saved to Wordcloud New & Emerging Terms.png")