import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from classes.stopwordsloader import StopwordsISO


class WordCloudGenerator:
    """
    A class to generate word clouds from text input.
    """

    def __init__(self, text=None, file_path=None, words_list=None, stopwords=None):
        """
        Initializes the WordCloudGenerator instance.

        Args:
            text (str, optional): Raw text to generate the word cloud.
            file_path (str, optional): Path to a text file to read from.
            words_list (list, optional): List of words to generate the word cloud.
            stopwords (set, optional): Set of stopwords to exclude.
        """
        sw = StopwordsISO()
        self.stopwords = sw.get_all_stopwords()
        self.text = self._load_text(text, file_path, words_list)

    def _load_text(self, text, file_path, words_list):
        """Loads and processes text based on input type."""
        if text:
            return text
        elif file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        elif words_list:
            return " ".join(words_list)
        else:
            return ""  # Default to empty text instead of raising an error

    def set_text(self, text=None, file_path=None, words_list=None):
        """
        Updates the text for word cloud generation.

        Args:
            text (str, optional): Raw text to update.
            file_path (str, optional): Path to a text file to read from.
            words_list (list, optional): List of words to update the text.
        """
        self.text = self._load_text(text, file_path, words_list)

    def generate_wordcloud(
        self,
        text=None,
        width=800,
        height=600,
        max_words=200,
        background_color="white",
        colormap="viridis",
    ) -> plt:
        """
        Generates and returns the word cloud plot.

        Args:
            text (str, optional): Temporary text override (does not modify stored text).
            width (int, optional): Width of the word cloud image.
            height (int, optional): Height of the word cloud image.
            max_words (int, optional): Maximum words to display.
            background_color (str, optional): Background color of the word cloud.
            colormap (str, optional): Colormap for visualization.

        Returns:
            matplotlib.pyplot: The generated word cloud plot.
        """
        wordcloud_text = text if text else self.text  # Use provided text or stored text
        if not wordcloud_text:
            raise ValueError("No text provided for word cloud generation.")

        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color=background_color,
            colormap=colormap,
            stopwords=self.stopwords,
        ).generate(wordcloud_text)

        plt.figure(figsize=(20, 10), facecolor="k")
        plt.imshow(wordcloud)
        plt.axis("off")
        return plt
