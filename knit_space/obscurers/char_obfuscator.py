# knit_space/obscurers/char_obfuscator.py
import random
import string
from typing import Dict, Tuple, List, Optional

class CharObfuscator:
    """
    Handles the creation of a character map and obfuscation of text
    for lowercase English letters.
    """
    ENGLISH_LOWERCASE = string.ascii_lowercase
    # Pool of symbols to map to. Ensure it has at least 26 unique characters.
    # Using a mix of Greek lowercase and some uppercase that are distinct.
    DEFAULT_SYMBOLS_POOL = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ" # 48 Greek chars

    def __init__(self, symbols_pool: Optional[str] = None):
        self.symbols_pool_str = symbols_pool if symbols_pool else self.DEFAULT_SYMBOLS_POOL
        self.unique_symbols_for_mapping = self._get_unique_symbols_from_pool()

        if len(self.unique_symbols_for_mapping) < 26:
            raise ValueError(
                f"Symbol pool must contain at least 26 unique characters. "
                f"Provided pool (after filtering unique) has {len(self.unique_symbols_for_mapping)}."
            )

    def _get_unique_symbols_from_pool(self) -> List[str]:
        """Ensures the symbols are unique characters."""
        return sorted(list(set(list(self.symbols_pool_str)))) # Sorted for deterministic behavior if pool is fixed

    def create_mapping(self) -> Dict[str, str]:
        """
        Creates a random, deterministic mapping from lowercase English letters
        to a unique subset of the symbols pool.
        The mapping is deterministic for a given shuffle of symbols_for_mapping.
        To get different maps for different test items, ensure this is called
        with a freshly shuffled list or re-shuffle self.unique_symbols_for_mapping
        before calling if that's the desired behavior for a single Obfuscator instance.

        For the MMLU test, we want a *new* random map per QAItem, so we'll
        shuffle the pool and pick 26 each time in the test class.
        This method demonstrates how a map *could* be fixed if the pool was fixed.
        """
        if len(self.unique_symbols_for_mapping) < 26:
             # This check should ideally be in __init__ more strictly or the pool expanded
            raise ValueError("Not enough unique symbols in the pool for a 26-letter mapping.")

        # For a new random map each time it's needed by the test:
        shuffled_symbols = random.sample(self.unique_symbols_for_mapping, k=26)
        
        char_map = dict(zip(self.ENGLISH_LOWERCASE, shuffled_symbols))
        return char_map

    def obfuscate_text(self, text: str, char_map: Dict[str, str]) -> str:
        """
        Obfuscates the input text by replacing lowercase English letters
        according to the provided character map. Other characters are unchanged.
        """
        obfuscated_chars = []
        for char in text:
            if 'a' <= char <= 'z':  # Check if it's a lowercase English letter
                obfuscated_chars.append(char_map.get(char, char))
            else:
                obfuscated_chars.append(char)
        return "".join(obfuscated_chars)

    def format_map_for_prompt(self, char_map: Dict[str, str]) -> str:
        """Formats the character map into a string for the LLM prompt."""
        # Sort by English letter for consistent display
        sorted_map_items = sorted(char_map.items())
        return ", ".join([f"{eng} → {greek}" for eng, greek in sorted_map_items])

if __name__ == '__main__':
    # Example Usage
    obfuscator = CharObfuscator()

    # Create a specific mapping for a test item
    current_char_map = obfuscator.create_mapping()
    print("Generated Character Map:")
    print(obfuscator.format_map_for_prompt(current_char_map))
    print("-" * 20)

    sample_text = "This is a Test for the MMLU dataset. Question 1: What is 2+2? Option A is four."
    print(f"Original Text:\n{sample_text}")

    obfuscated_sample = obfuscator.obfuscate_text(sample_text, current_char_map)
    print(f"\nObfuscated Text:\n{obfuscated_sample}")
    print("-" * 20)

    # Demonstrate that calling create_mapping again can produce a different map
    # (due to random.sample)
    new_char_map = obfuscator.create_mapping()
    print("Newly Generated Character Map (should differ if pool is large enough):")
    print(obfuscator.format_map_for_prompt(new_char_map))
    obfuscated_again = obfuscator.obfuscate_text(sample_text, new_char_map)
    print(f"\nObfuscated Text with new map:\n{obfuscated_again}")