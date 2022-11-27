from wordcloud import WordCloud
import matplotlib.pyplot as plt

def positive_word_cloud(text, text_column):
    positives = text.query("classification == 1")
    all_words = ' '.join([text for text in positives[text_column]])
    word_cloud = WordCloud(
        width=800, height=500,
        max_font_size=110,
        collocations=False
    ).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def negative_word_cloud(text, text_column):
    negatives = text.query("classification == 0")
    all_words = ' '.join([text for text in negatives[text_column]])
    word_cloud = WordCloud(
        width=800, height=500,
        max_font_size=110,
        collocations=False
    ).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()