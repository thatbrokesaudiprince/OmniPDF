import json
from typing import Set, Union, Iterable


class StopwordsISO:
    """
    A class to manage stopwords for multiple languages based on ISO 639-1 language codes.
    """

    def __init__(self, filepath: str = "static/stopwords-iso.json"):
        """
        Initializes the StopwordsISO instance by loading stopwords from a JSON file.

        Parameters
        ----------
        filepath : str, optional
            Path to the stopwords JSON file, by default 'frontend/static/stopwords-iso.json'
        """
        self._stopwords_all = self._load_stopwords(filepath)
        self._langs = set(self._stopwords_all.keys())

    @staticmethod
    def _load_stopwords(filepath: str) -> dict:
        """Loads the stopwords JSON file."""
        with open(filepath, encoding="utf-8") as json_data:
            return json.load(json_data)

    def langs(self) -> Set[str]:
        """Returns a set of supported languages in ISO 639-1 format."""
        return self._langs

    def has_lang(self, lang: str) -> bool:
        """Checks if stopwords exist for a given language."""
        return lang in self._langs

    def stopwords(self, langs: Union[str, Iterable[str]]) -> Set[str]:
        """
        Retrieves a set of stopwords for the requested language(s).

        Parameters
        ----------
        langs : str or Iterable[str]
            A language code or a collection of language codes in ISO 639-1 format.

        Returns
        -------
        Set[str]
            A set containing all stopwords for the requested languages.
        """
        words = set()

        if isinstance(langs, str):
            if self.has_lang(langs):
                words.update(self._stopwords_all[langs])
        else:
            try:
                for lang in langs:
                    if self.has_lang(lang):
                        words.update(self._stopwords_all[lang])
            except TypeError:
                print("'langs' must be a string or an iterable of strings.")

        return words

    def get_all_stopwords(self) -> Set[str]:
        """Returns a set of all stopwords across all supported languages."""
        words = set()
        for lang in self._langs:
            words.update(self._stopwords_all[lang])
        return words
