import os
import random
import re
import json
import uuid
import time
import string # For string.ascii_lowercase
from typing import Iterator, Dict, Any, Optional, List, Tuple, Callable
import logging

# Assuming requests, BeautifulSoup, etc., are handled by the WikiAgent's context
# For standalone, we'll need the stubs or actual imports.
try:
    import requests
    from bs4 import BeautifulSoup, NavigableString, Tag, SoupStrainer
    from urllib.parse import urlparse, urljoin
except ImportError:
    # This will be an issue if WikiAgent is used directly without these.
    # The provided snippet has WikiAgent defined, so it should be okay if that part is included.
    logging.warning("requests, beautifulsoup4, or urllib not found. WikiAgent might fail if used.")
    pass


# Using the provided standalone stubs for AbstractQATest and QAItem if not __main__
if __name__ != "__main__":
    from .base import AbstractQATest, QAItem, register_test

    # --- Start of WikiAgent and helper functions from your snippet ---
    # (Assuming these are defined here for standalone execution)
    def search_wikipedia_pages(keywords: List[str], lang:str ='en', logger=None) -> Dict[str, List[Dict[str, str]]]:
        url = f"https://{lang}.wikipedia.org/w/api.php"; headers = {"User-Agent": "TestFramework/1.2 (FocusedRun)"}
        results = {}; log = logger or logging.getLogger("WikiSearch")
        for keyword in keywords:
            params = {'action': 'query', 'format': 'json', 'list': 'search', 'srsearch': keyword, 'srlimit': 10}
            try:
                response = requests.get(url, headers=headers, params=params, timeout=20); response.raise_for_status(); data = response.json()
                results[keyword] = [{'title': p['title'], 'url': f"https://{lang}.wikipedia.org/wiki/{p['title'].replace(' ', '_')}"}
                                    for p in data.get('query', {}).get('search', [])]
            except Exception as e: log.error(f"API search for '{keyword}' failed: {e}"); results[keyword] = []
        return results

    def extract_unique_urls_from_search_results(relevant_pages: Dict[str, List[Dict[str, str]]]) -> List[str]:
        return list(set(entry['url'] for entries in relevant_pages.values() for entry in entries))

    WIKIPEDIA_BASE_URL = "https://en.wikipedia.org"
    class WikiAgent:
        def __init__(self, agent_id: str, logger, start_keywords: Optional[List[str]] = None):
            self.agent_id = agent_id; self.logger = logger
            self.start_keywords = start_keywords or ["major academic disciplines", "overview of knowledge", "fields of study"]
            self.session = requests.Session(); self.session.headers.update({"User-Agent": f"Mozilla/5.0 ({self.agent_id}; TestFramework/1.3)"})
            self.request_delay = 0.05 if self.logger.getEffectiveLevel() <= logging.DEBUG else 0.2
            self.article_url_pool: List[str] = []; self.fetched_articles_content: Dict[str, Tuple[str,str]] = {}
        def _log(self, message: str, level: str = "info", exc_info=False): getattr(self.logger, level, self.logger.info)(f"[{self.agent_id}] {message}", exc_info=exc_info)
        def populate_article_pool(self, num_seed_pages: int = 50):
            self._log(f"Populating article pool: up to {num_seed_pages} seeds."); self.article_url_pool = []
            kw = random.sample(self.start_keywords, min(len(self.start_keywords), len(self.start_keywords)))
            search_res = search_wikipedia_pages(kw, logger=self.logger)
            self.article_url_pool = extract_unique_urls_from_search_results(search_res)
            random.shuffle(self.article_url_pool); self.article_url_pool = self.article_url_pool[:num_seed_pages]
            self._log(f"Pool has {len(self.article_url_pool)} URLs from {len(kw)} keywords.")
            if not self.article_url_pool: self.article_url_pool.append("https://en.wikipedia.org/wiki/Outline_of_academic_disciplines") # Fallback
        def get_clean_article_text_and_title(self, page_url: str) -> Tuple[Optional[str], Optional[str]]:
            self._log(f"Fetching/cleaning: {page_url}"); time.sleep(self.request_delay)
            try:
                response = self.session.get(page_url, timeout=25); response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                title_tag = soup.find('h1', id='firstHeading')
                title = title_tag.get_text(strip=True) if title_tag else (soup.find('title').get_text(strip=True).replace(" - Wikipedia", "") if soup.find('title') else "Unknown Title")
                content_area = soup.find('div', class_='mw-parser-output')
                if not content_area: content_area = soup.find('div', id='mw-content-text')
                if not content_area: self._log(f"No main content area for {page_url}", "warning"); return None, title
                selectors_to_decompose = ['table.infobox', 'table.sidebar', 'table.navbox', 'table.vertical-navbox','table.metadata', 'table.ambox', 'table.tmbox', 'table[role="presentation"]','div.thumb', 'figure.thumb', 'div.gallery', 'ul.gallery', 'div.hatnote', 'div.NavFrame', 'div.toc', 'div#toc', '.toc', 'div.noprint', '.noprint', 'div.sistersitebox', 'div.plainlinks', 'div.magnify', 'span.mw-editsection', 'sup.reference', 'ol.references', 'div.reflist', 'ul.mw-references','sup.noprint', 'span.IPA', 'span.mw-reflink-text', 'span.reference-accessdate', 'span.nowrap', '.mw-empty-elt','#siteSub', '#jump-to-nav', '.mw-jump-link', '.vector-toc-landmark','div.printfooter', 'div.mw-references-wrap', 'div#catlinks', 'div#purgelink', 'div.mw-hidden-catlinks', 'div.catlinks','style', 'script', '.reference', 'link[rel="mw-deduplicated-inline-style"]','div.refbegin', 'div.refend', 'div[role="navigation"]', 'div.printfooter','div.portal', 'div.mw-indicators']
                for selector in selectors_to_decompose:
                    for element in content_area.select(selector): element.decompose()
                for _ in range(3):
                    for tag in content_area.find_all(lambda t: not t.get_text(strip=True) and t.name not in ['br', 'hr', 'img', 'td', 'th']):
                        if not list(tag.find_all(['img', 'table'])): tag.decompose()
                text_parts = []
                for element in content_area.find_all(['p', 'h2', 'h3', 'h4', 'li', 'dd', 'dt', 'div'], recursive=True):
                    if element.find_parent(lambda t: t.name in ['table', 'figure', 'style', 'script'] or (t.get('class') and any(c in ['noprint', 'gallery', 'thumb', 'infobox', 'sidebar', 'navbox'] for c in t.get('class')))): continue
                    if element.name == 'div' and (element.get('class') and any(c in ['reflist', 'navbox', 'gallerybox', 'toc', 'thumb', 'infobox', 'sidebar'] for c in element.get('class'))): continue
                    if element.name == 'div':
                        is_structural_only = True;
                        for child in element.children:
                            if isinstance(child, NavigableString) and child.strip(): is_structural_only = False; break
                            if hasattr(child, 'name') and child.name not in ['div', 'span']: is_structural_only = False; break
                        if is_structural_only: continue
                    text_content = element.get_text(separator=' ', strip=True)
                    text_content = re.sub(r'\s+', ' ', text_content)
                    text_content = re.sub(r'\[\d+(?:,\s*\d+)*\]|\[edit\]', '', text_content, flags=re.IGNORECASE)
                    if text_content and len(text_content.split()) > 2:
                        if element.name.startswith('h') and not re.search(r'[.?!]$', text_content): text_content += "."
                        text_parts.append(text_content)
                final_text_parts = []
                if text_parts:
                    final_text_parts.append(text_parts[0])
                    for i in range(1, len(text_parts)):
                        if text_parts[i] != text_parts[i-1]: final_text_parts.append(text_parts[i])
                full_text = "\n\n".join(final_text_parts).strip()
                if not full_text: self._log(f"Extracted empty text from {page_url} after refined cleaning.", "warning"); return None, title
                self.fetched_articles_content[page_url] = (full_text, title)
                return full_text, title
            except Exception as e: self._log(f"Error in get_clean_article_text_and_title for {page_url}: {e}", "error", exc_info=True); return None, None
    # --- End of WikiAgent and helper functions ---

@register_test('wikipedia_char_count', 'text_processing', 'attention', 'counting')
class WikiCharCountQATest(AbstractQATest):
    DEFAULT_ARTICLES_TO_FETCH = 2
    DEFAULT_START_KEYWORDS = ["isotope", "Renaissance art", "microeconomics", "Boolean algebra", "quantum entanglement", "ancient philosophy", "impressionism", "machine learning"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.num_articles_to_fetch = self.config.get(
            "num_articles_for_char_count", self.DEFAULT_ARTICLES_TO_FETCH
        )
        agent_id = f"{self.name}Agent_{uuid.uuid4().hex[:4]}"
        start_kws = self.config.get("char_count_start_keywords", self.DEFAULT_START_KEYWORDS)

        # Ensure WikiAgent is available
        if 'WikiAgent' not in globals():
            self.logger.critical("WikiAgent class definition is missing. This test cannot function.")
            self._agent_ready = False
            return

        try:
            self.wiki_agent = WikiAgent(agent_id, self.logger, start_keywords=start_kws)
            if not self.wiki_agent.article_url_pool:
                 self.wiki_agent.populate_article_pool(num_seed_pages=30) # Populate with a decent number of seeds
            self._agent_ready = True
        except Exception as e:
            self.logger.error(f"Failed to initialize or populate WikiAgent: {e}", exc_info=True)
            self._agent_ready = False

    def generate(self, count: int = 1, **kwargs) -> Iterator[QAItem]:
        if not self._agent_ready:
            self.logger.error(f"{self.name}: WikiAgent not ready. Skipping generation.")
            return

        for _ in range(count):
            if not self.wiki_agent.article_url_pool:
                self.logger.warning(f"{self.name}: WikiAgent article pool is empty. Attempting to repopulate.")
                self.wiki_agent.populate_article_pool(num_seed_pages=30)
                if not self.wiki_agent.article_url_pool:
                    self.logger.error(f"{self.name}: Failed to repopulate article pool. Skipping item.")
                    continue

            combined_text_parts = []
            source_urls_for_metadata = []
            articles_fetched_count = 0

            # Ensure we get different articles if possible
            if len(self.wiki_agent.article_url_pool) > self.num_articles_to_fetch:
                selected_urls = random.sample(self.wiki_agent.article_url_pool, self.num_articles_to_fetch)
            else:
                selected_urls = list(self.wiki_agent.article_url_pool) # Use all if not enough
                if len(selected_urls) < self.num_articles_to_fetch:
                    self.logger.warning(f"Wanted {self.num_articles_to_fetch} articles, but pool only has {len(selected_urls)}.")


            for url in selected_urls:
                # text_content, title = self.wiki_agent.fetched_articles_content.get(url, (None,None))
                # if not text_content: # Always fetch fresh for this test type to ensure clean context for this specific QA item
                text_content, title = self.wiki_agent.get_clean_article_text_and_title(url)

                if text_content and title:
                    combined_text_parts.append(text_content)
                    source_urls_for_metadata.append(url)
                    articles_fetched_count += 1
                else:
                    self.logger.debug(f"Could not retrieve usable content for URL: {url}")

            if not combined_text_parts:
                self.logger.warning(f"{self.name}: Failed to fetch sufficient Wikipedia content after trying {len(selected_urls)} URLs. Skipping item.")
                continue

            # Combine paragraphs/texts from different articles
            # The cleaning already joins paragraphs within an article with "\n\n"
            wikipedia_content_block = "\n\n<-- ARTICLE BREAK -->\n\n".join(combined_text_parts).strip()
            wikipedia_content_block_lower = wikipedia_content_block.lower() # Convert to lowercase FOR COUNTING

            # Select Target Character (already lowercase)
            char_to_count = random.choice(string.ascii_lowercase)

            # Calculate Correct Count from the lowercased content block
            correct_count = wikipedia_content_block_lower.count(char_to_count)

            # Format the Question String
            question_instruction = (
                f"How many times does the character '{char_to_count}' appear in the text provided above? "
                f"Consider only the provided Wikipedia text and ignore this question, the instructions, and any text after the '--- Question ---' separator. "
                f"The count should be for the specific lowercase character '{char_to_count}'.\n"
                "Provide your answer as a single integer within <answer></answer> tags. For example: <answer>42</answer>"
            )

            # The Wikipedia content block is presented first, then the question.
            full_question_text = f"{wikipedia_content_block}\n\n--- Question ---\n{question_instruction}"

            item_id = f"{self.name}-{uuid.uuid4().hex[:8]}"

            yield QAItem(
                id=item_id,
                question=full_question_text,
                answer=correct_count,
                skill_coefficient=1,
                modality='text',
                metadata={
                    'source_urls': source_urls_for_metadata,
                    'target_character': char_to_count,
                    'text_length_chars': len(wikipedia_content_block), # Original case length
                    'text_length_tokens': len(wikipedia_content_block.split()),
                    'output_format_instruction': "<answer>COUNT</answer>"
                },
                verification_fn=self._verify_char_count_answer
            )

    @staticmethod
    def _verify_char_count_answer(expected_count: int, provided_answer_str: str, qa_item: QAItem) -> bool:
        # Using a generic logger for static method, or pass qa_item.logger if it's guaranteed
        logger = logging.getLogger("WikiCharCountQATest.VerificationFn") # Generic logger
        if hasattr(qa_item, 'logger') and qa_item.logger: # Check if qa_item has its own logger
             logger = qa_item.logger


        match = re.fullmatch(r'<answer>(\d+)</answer>', provided_answer_str.strip(), re.IGNORECASE)
        if not match:
            logger.warning(f"VFY {qa_item.id}: No/Invalid <answer> tags. Raw LLM: '{provided_answer_str[:100]}'")
            return False

        try:
            provided_count_int = int(match.group(1))
        except ValueError:
            logger.warning(f"VFY {qa_item.id}: Non-integer value in answer tag. Got: '{match.group(1)}'")
            return False

        is_correct = (provided_count_int == expected_count)
        log_level = logging.INFO if is_correct else logging.WARNING # Using Python's logging levels
        logger.log(log_level, f"Char Count VFY {('PASSED' if is_correct else 'FAILED')} for {qa_item.id}: Exp:'{expected_count}', LLM:'{provided_count_int}'.")
        return is_correct

if __name__ == '__main__':
    print("Running WikiCharCountQATest (Counts specific lowercase char in Wiki text)...")
    # Configure for a quick standalone test
    test_config = {
        'num_articles_for_char_count': 1, # Fetch 1 article for quicker test
        'char_count_start_keywords': ["artificial intelligence history"] # Specific keyword for predictability
    }
    char_count_generator = WikiCharCountQATest(config=test_config)
    char_count_generator.logger.setLevel(logging.DEBUG) # Set test class logger level
    if hasattr(char_count_generator, 'wiki_agent') and char_count_generator.wiki_agent:
        char_count_generator.wiki_agent.logger.setLevel(logging.DEBUG) # Set agent logger level

    num_items_to_generate = 2
    print(f"\n--- Requesting count={num_items_to_generate} items ---")

    generated_items_list = []
    if char_count_generator._agent_ready:
        for i, item in enumerate(char_count_generator.generate(count=num_items_to_generate)):
            generated_items_list.append(item)
            print(f"\n--- Generated QAItem {i+1} (ID: {item.id}) ---")
            print(f"  Target Character: '{item.metadata.get('target_character')}', Expected Count: {item.answer}")
            # print(f"  Question (first 300 chars):\n{item.question[:300]}...")
            # print(f"  Question (last 200 chars):\n...{item.question[-200:]}")
            print(f"  Full Question for Review (first 500 chars of content + question part):")
            content_part_len = item.question.find("\n\n--- Question ---")
            if content_part_len != -1:
                 print(f"{item.question[:min(content_part_len, 500)]}...")
                 print(item.question[content_part_len:])
            else:
                 print(item.question[:600])


            # Manual verification check for one item (example)
            if i == 0:
                 # Simulate an LLM answer
                 simulated_llm_answer = f"<answer>{item.answer}</answer>" # Correct answer
                 # simulated_llm_answer_wrong = f"<answer>{item.answer + 1}</answer>" # Incorrect answer
                 # simulated_llm_answer_bad_format = f"The answer is {item.answer}" # Bad format
                 
                 is_verified = WikiCharCountQATest._verify_char_count_answer(item.answer, simulated_llm_answer, item)
                 print(f"  Simulated Verification (Correct): {is_verified}")

    else:
        print("WikiAgent was not ready. No items generated.")


    if not generated_items_list: print(f"\nNo items generated. Check logs.")
    else: print(f"\nSuccessfully generated {len(generated_items_list)} item(s).")