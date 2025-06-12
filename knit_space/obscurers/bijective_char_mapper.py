import random
import string
import os

class BijectiveCharacterMapper:
    GREEK_CHARS = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
    DEVANAGARI_CHARS = "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
    CYRILLIC_CHARS = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    SOURCE_CHARS = string.ascii_lowercase + string.digits

    TOKEN_INTERNAL_DELIM = "_"
    ENCODING_DELIM = "%%TOK%%"
    MAPPING_FILE_SEPARATOR = "\t"
    DEFAULT_MAPPING_FILE = "persistent_char_mapping.txt"

    def __init__(self, mapping_file_path: str = DEFAULT_MAPPING_FILE):
        self.mapping_file_path = mapping_file_path
        self._char_to_token: dict[str, str] = {}
        self._token_to_char: dict[str, str] = {}

        if not self._load_mappings():
            self._generate_mappings()
            self._save_mappings()

    def _generate_token(self) -> str:
        part1 = random.choice(self.GREEK_CHARS)
        part2 = random.choice(self.DEVANAGARI_CHARS)
        part3 = random.choice(self.CYRILLIC_CHARS)
        return f"{part1}{self.TOKEN_INTERNAL_DELIM}{part2}{self.TOKEN_INTERNAL_DELIM}{part3}"

    def _generate_mappings(self):
        self._char_to_token.clear()
        self._token_to_char.clear()
        used_tokens = set()

        for char_to_map in self.SOURCE_CHARS:
            while True:
                token = self._generate_token()
                if token not in used_tokens:
                    used_tokens.add(token)
                    self._char_to_token[char_to_map] = token
                    self._token_to_char[token] = char_to_map
                    break
        print(f"Generated {len(self._char_to_token)} new character mappings.")

    def _save_mappings(self):
        try:
            with open(self.mapping_file_path, "w", encoding="utf-8") as f:
                for char, token in self._char_to_token.items():
                    f.write(f"{char}{self.MAPPING_FILE_SEPARATOR}{token}\n")
            print(f"Character mappings saved to '{self.mapping_file_path}'.")
        except IOError as e:
            print(f"Error saving mappings to file '{self.mapping_file_path}': {e}")

    def _load_mappings(self) -> bool:
        if not os.path.exists(self.mapping_file_path):
            return False

        temp_char_to_token = {}
        temp_token_to_char = {}
        try:
            with open(self.mapping_file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line: continue
                    parts = line.split(self.MAPPING_FILE_SEPARATOR, 1)
                    if len(parts) == 2:
                        char, token = parts[0], parts[1]
                        if char not in self.SOURCE_CHARS or token in temp_token_to_char or char in temp_char_to_token:
                            print(f"Invalid or duplicate entry in mapping file (line {i+1}). Will regenerate.")
                            return False
                        temp_char_to_token[char] = token
                        temp_token_to_char[token] = char
                    else:
                        print(f"Malformed line in mapping file (line {i+1}). Will regenerate.")
                        return False
            
            if not temp_char_to_token or set(temp_char_to_token.keys()) != set(self.SOURCE_CHARS):
                print("Mapping file incomplete or empty. Will regenerate.")
                return False

            self._char_to_token = temp_char_to_token
            self._token_to_char = temp_token_to_char
            print(f"Character mappings loaded from '{self.mapping_file_path}'.")
            return True
        except Exception as e:
            print(f"Error loading or parsing mapping file '{self.mapping_file_path}': {e}. Will regenerate.")
            return False

    def get_mapping_table(self) -> dict:
        return self._char_to_token.copy()

    def get_reverse_mapping_table(self) -> dict:
        return self._token_to_char.copy()

    def encode(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")
        
        encoded_parts = []
        for original_char in text:
            char_to_check = original_char.lower()
            if char_to_check in self._char_to_token:
                encoded_parts.append(self._char_to_token[char_to_check])
            else:
                encoded_parts.append(original_char)
        return self.ENCODING_DELIM.join(encoded_parts)

    def decode(self, encoded_text: str) -> str:
        if not isinstance(encoded_text, str):
             raise TypeError("Input must be a string.")
        if not encoded_text: return ""

        parts = encoded_text.split(self.ENCODING_DELIM)
        decoded_elements = []
        for part in parts:
            decoded_elements.append(self._token_to_char.get(part, part))
        return "".join(decoded_elements)

if __name__ == "__main__":
    # Ensure a clean slate for demo if needed
    if os.path.exists(BijectiveCharacterMapper.DEFAULT_MAPPING_FILE):
         os.remove(BijectiveCharacterMapper.DEFAULT_MAPPING_FILE)

    mapper = BijectiveCharacterMapper()
    
    original = "Sample Text 123 for testing!"
    print(f"Original: '{original}'")

    encoded = mapper.encode(original)
    print(f"Encoded:  '{encoded}'")

    decoded = mapper.decode(encoded)
    print(f"Decoded:  '{decoded}'")

    normalized_original = "".join([c.lower() if c.lower() in BijectiveCharacterMapper.SOURCE_CHARS else c for c in original])
    print(f"Match (normalized original vs decoded): {normalized_original == decoded}")

    mapper2 = BijectiveCharacterMapper()
    encoded2 = mapper2.encode(original)
    assert encoded == encoded2, "Mappings did not persist correctly!"
    print("Persistence test passed.")