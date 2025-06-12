import os
import random
import re
import json
import uuid
import time
from typing import Iterator, Dict, Any, Optional, List, Tuple, Callable
import logging

try:
    import requests
    from bs4 import BeautifulSoup, NavigableString, Tag, SoupStrainer
    from urllib.parse import urlparse, urljoin
except ImportError:
    raise ImportError("requests, beautifulsoup4, and urllib are required.")

if __name__ != "__main__":
    from .base import AbstractQATest, QAItem, register_test
else:
    logging.basicConfig(level=logging.DEBUG)
    print("Running in standalone mode.")
    class AbstractQATest:
        def __init__(self, config=None):
            self.config = config or {}; self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.hasHandlers():
                h = logging.StreamHandler(); h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(h); self.logger.propagate = False
            self.logger.setLevel(logging.DEBUG)
        @property
        def name(self): return self.__class__.__name__
    class QAItem:
        def __init__(self, id, question, answer, modality, metadata, verification_fn=None):
            self.id=id;self.question=question;self.answer=answer;self.modality=modality;self.metadata=metadata;self.verification_fn=verification_fn
            self.logger = logging.getLogger("QAItemStandalone")
            if not self.logger.hasHandlers():
                h = logging.StreamHandler(); h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(h); self.logger.propagate = False
            self.logger.setLevel(logging.DEBUG)
    def register_test(*args, **kwargs):
        def decorator(cls): return cls
        return decorator

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
        if not self.article_url_pool: self.article_url_pool.append("https://en.wikipedia.org/wiki/Outline_of_academic_disciplines")
    def get_clean_article_text_and_title(self, page_url: str) -> Tuple[Optional[str], Optional[str]]:
        self._log(f"Fetching/cleaning: {page_url}"); time.sleep(self.request_delay)
        try:
            response = self.session.get(page_url, timeout=25); response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser') # Changed back to html.parser for fewer dependencies
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
                    is_structural_only = True
                    for child in element.children:
                        if isinstance(child, NavigableString) and child.strip(): is_structural_only = False; break
                        if child.name not in ['div', 'span']: is_structural_only = False; break
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
    def build_book_from_pool(self, target_token_count: int) -> Tuple[Optional[str], int]:
        if not self.article_url_pool: self.populate_article_pool()
        if not self.article_url_pool: self._log("Pool empty.", "error"); return None, 0
        bp = []; ct = 0; auc = 0; au = random.sample(self.article_url_pool, len(self.article_url_pool))
        for url in au:
            if ct >= target_token_count and auc > 0 : break
            full_text, title = self.fetched_articles_content.get(url, (None, None))
            if not full_text: full_text, title = self.get_clean_article_text_and_title(url)
            if full_text and title:
                at = count_tokens(full_text)
                if at < 100 and auc > 0 : self._log(f"Skip short '{title}' ({at} tokens)."); continue
                bp.append(f"\n\n--- Article Start: {title} (URL: {url}) ---\n{full_text}\n--- Article End: {title} ---")
                ct += at; auc += 1; self._log(f"Added '{title}' ({at} tokens). Book at {ct} tokens.")
            else: self._log(f"No usable text from {url}.")
        if not bp: self._log("Failed to assemble content.", "warning"); return None, 0
        fbt = "".join(bp).strip(); act_tok = count_tokens(fbt)
        self._log(f"Book built: Target ~{target_token_count}, Actual {act_tok} tokens, {auc} articles.")
        return fbt, act_tok

def count_tokens(text: str) -> int: return len(text.split())
def get_sentences(text: str) -> List[str]:
    text = re.sub(r'\s*\n\s*', ' ', text)
    s = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|!)\s', text)
    return [st.strip() for st in s if st.strip() and len(st.split()) > 1]
def corrupt_sentence(sentence: str, method: str = "shuffle") -> Tuple[str, str, Optional[str], Optional[str]]:
    words = sentence.split(); osg: Optional[str] = None; csg: Optional[str] = None; cs = ""; cd = method
    if len(words) < 3 and method == "shuffle": return sentence, "no_change_too_short_for_shuffle", None, None
    if not words and method != "add_word": return sentence, "no_change_empty_sentence", None, None
    if method == "shuffle":
        lw = words[-1]; p = ""
        if re.search(r'[.?!]$', lw) and len(lw) > 1: p = lw[-1]; words[-1] = lw[:-1]
        elif re.search(r'[.?!]$', lw) and len(lw) == 1: pass 
        sw = words[:]; otw = tuple(words)
        if len(sw) > 1:
            for _ in range(10): 
                random.shuffle(sw)
                if tuple(sw) != otw: break 
            if tuple(sw) == otw: return sentence, "no_change_shuffle_failed_to_alter", None, None
        cs = " ".join(sw);
        if p and not cs.endswith(p): cs += p
        osg = " ".join(words); csg = cs; cd = f"words_shuffled"
    elif method == "replace_word":
        if not words: return sentence, "no_change_empty_sentence_for_replace", None, None
        idx = random.randrange(len(words)); osg = words[idx]
        if len(osg) < 3 and not osg.isalnum(): return sentence, "no_change_word_too_short_or_punctuation_for_replace", None, None
        rc = "abcdefghijklmnopqrstuvwxyz"
        if osg.istitle(): rw = "Repl" + ''.join(random.choices(rc, k=random.randint(2,5)))
        elif osg.isupper(): rw = "REPL" + ''.join(random.choices(rc.upper(), k=random.randint(2,5)))
        else: rw = "repl" + ''.join(random.choices(rc, k=random.randint(2,5)))
        while rw.lower() == osg.lower(): 
            ac = "xzywvutsrqponmlkjihgfe"
            if osg.istitle(): rw = "Alt" + ''.join(random.choices(ac, k=random.randint(2,5)))
            elif osg.isupper(): rw = "ALT" + ''.join(random.choices(ac.upper(), k=random.randint(2,5)))
            else: rw = "alt" + ''.join(random.choices(ac, k=random.randint(2,5)))
        temp_words = words[:]; temp_words[idx] = rw; cs = " ".join(temp_words); csg = rw; cd = f"word_replaced ('{osg}' with '{csg}')"
    elif method == "delete_word":
        if not words: return sentence, "no_change_empty_sentence_for_delete", None, None
        idx = random.randrange(len(words)); 
        temp_words = words[:]; osg = temp_words.pop(idx)
        if not temp_words: return sentence, "no_change_deleted_only_word", None, None
        cs = " ".join(temp_words); csg = None; cd = f"word_deleted ('{osg}')"
    else: return corrupt_sentence(sentence, "shuffle") 
    return cs, cd, osg, csg

@register_test('long_context', 'wikipedia', 'find_changed_word', 'in_memory_content_full_prompt')
class LongContextWikiBookTest(AbstractQATest):
    DEFAULT_TOKEN_TARGETS = [1000, 10000, 100000] 
    DEFAULT_START_KEYWORDS_FOR_POOL = ["natural science", "social science", "formal science", "applied science", "humanities", "history of ideas", "list of academic fields", "overview of philosophy", "introduction to physics", "concepts in biology"]
    CORRUPTION_METHODS_FOR_WORD_FIND = ["replace_word", "delete_word"]
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.wiki_agent = WikiAgent(f"{self.name}Agent", self.logger, start_keywords=self.config.get("long_context_start_keywords", self.DEFAULT_START_KEYWORDS_FOR_POOL))
        self.token_targets = self.config.get("long_context_token_targets", self.DEFAULT_TOKEN_TARGETS)
        self._book_cache: Dict[str, Tuple[Optional[str], int]] = {}
    @property
    def supported_modalities(self) -> List[str]: return ['text']
    def _get_or_create_book_text(self, target_tokens: int, book_version_identifier: str, force_regenerate: bool = False) -> Tuple[Optional[str], int]:
        cache_key = f"{target_tokens}_{book_version_identifier}"
        if not force_regenerate and cache_key in self._book_cache:
            cached_text, cached_tokens = self._book_cache[cache_key]
            if cached_text and cached_tokens > target_tokens * 0.05: self.logger.debug(f"Using cached book for {cache_key}"); return cached_text, cached_tokens
        self.logger.info(f"Generating book: ~{target_tokens} tokens, version '{book_version_identifier}'.")
        if force_regenerate or not self.wiki_agent.article_url_pool or not book_version_identifier.endswith("_cached_pool"): 
             self.wiki_agent.fetched_articles_content.clear(); self.wiki_agent.populate_article_pool(num_seed_pages=max(25, target_tokens // 500))
        book_text, actual_tokens = self.wiki_agent.build_book_from_pool(target_tokens)
        if book_text and actual_tokens >= target_tokens * 0.1: self._book_cache[cache_key] = (book_text, actual_tokens); return book_text, actual_tokens
        else: self.logger.error(f"Failed book {target_tokens} v'{book_version_identifier}' (got {actual_tokens} tokens)."); self._book_cache[cache_key] = (None, 0); return None, 0
    @staticmethod
    def _verify_corrupted_word_tagged(expected_raw_word: str, provided_llm_answer_tagged: str, qa_item: QAItem) -> bool:
        logger = qa_item.logger if hasattr(qa_item, 'logger') else logging.getLogger("VerificationFnWordTagged")
        llm_word_extracted = None
        match = re.search(r"<answer>(.*?)</answer>", provided_llm_answer_tagged, re.IGNORECASE | re.DOTALL)
        if match: llm_word_extracted = match.group(1).strip()
        else: logger.warning(f"VFY {qa_item.id}: No <answer> tags. Raw LLM: '{provided_llm_answer_tagged[:150]}'"); return False
        norm_expected = re.sub(r'[^\w\s-]', '', str(expected_raw_word).strip().lower())
        norm_provided = re.sub(r'[^\w\s-]', '', str(llm_word_extracted).strip().lower())
        if not norm_expected: logger.error(f"VFY {qa_item.id}: Expected word empty. Original: '{expected_raw_word}'"); return False
        is_correct = norm_expected == norm_provided
        log_level = logging.INFO if is_correct else logging.WARNING
        logger.log(log_level, f"Word VFY (Tagged) {('PASSED' if is_correct else 'FAILED')} for {qa_item.id}: Exp(n):'{norm_expected}', Extracted LLM(n):'{norm_provided}'. Raw LLM:'{provided_llm_answer_tagged[:100]}'")
        return is_correct

    def generate(self, count: int = 1, difficulty: Optional[str] = None, **kwargs) -> Iterator[QAItem]:
        force_regenerate_all_books = kwargs.get('force_regenerate_books', False)
        if force_regenerate_all_books: self._book_cache.clear(); self.wiki_agent.fetched_articles_content.clear(); self.logger.info("Caches cleared.")
        num_sets_to_generate = count 
        self.logger.info(f"{self.name}: Requested {num_sets_to_generate} set(s) of {len(self.token_targets)} context-length items each.")
        total_items_yielded = 0
        for set_idx in range(num_sets_to_generate):
            current_set_num = set_idx + 1
            self.logger.info(f"Starting generation for set {current_set_num}/{num_sets_to_generate}.")
            force_regen_for_this_set_books = force_regenerate_all_books or (set_idx > 0) 
            if force_regen_for_this_set_books and not force_regenerate_all_books : 
                self.wiki_agent.article_url_pool = [] 
                self.wiki_agent.fetched_articles_content.clear()
                self.logger.debug(f"Cleared agent caches for new set {current_set_num}.")
            for target_tokens in self.token_targets:
                book_version_id = f"set{current_set_num}" 
                if not force_regen_for_this_set_books: book_version_id += "_cached_pool"
                self.logger.debug(f"  Processing target {target_tokens} tokens for set {current_set_num}.")
                original_book_text, actual_book_tokens = self._get_or_create_book_text(target_tokens, book_version_id, force_regen_for_this_set_books)
                if not original_book_text: self.logger.warning(f"    Skipping {target_tokens} for set {current_set_num}: book generation failed."); continue
                sentences = get_sentences(original_book_text)
                if not sentences or len(sentences) < 3: self.logger.warning(f"    Book for {target_tokens}, set {current_set_num} has <3 sentences. Skipping."); continue
                eligible_indices = [i for i, s in enumerate(sentences) if 5 < len(s.split()) < 80]
                if not eligible_indices: eligible_indices = [i for i, s in enumerate(sentences) if 3 < len(s.split()) < 120]
                if not eligible_indices: self.logger.error(f"    No eligible sentences for {target_tokens}, set {current_set_num}."); continue
                actual_original_sentence = sentences[random.choice(eligible_indices)]
                corruption_method = random.choice(self.CORRUPTION_METHODS_FOR_WORD_FIND)
                actual_corrupted_sentence, desc, orig_seg, corr_seg = corrupt_sentence(actual_original_sentence, corruption_method)
                if "no_change" in desc or orig_seg is None: self.logger.warning(f"    Corruption failed for {target_tokens}, set {current_set_num} ('{desc}'). Skipping."); continue
                try:
                    found_at = original_book_text.find(actual_original_sentence)
                    if found_at == -1: 
                        self.logger.error(f"    Original sentence not precisely found for {target_tokens}, set {current_set_num}. Sentence: '{actual_original_sentence[:50]}...' Attempting fallback search with normalized spaces.")
                        normalized_original_sentence = re.sub(r'\s+', ' ', actual_original_sentence)
                        normalized_book_text = re.sub(r'\s+', ' ', original_book_text)
                        found_at_norm = normalized_book_text.find(normalized_original_sentence)
                        if found_at_norm != -1:
                            # This is tricky as original_book_text has original newlines, etc.
                            # For now, if direct find fails, we skip to avoid complex reconstruction issues.
                            self.logger.error("    Normalized search found, but exact reconstruction is complex. Skipping item.")
                            continue
                        else:
                            self.logger.error("    Still not found even after normalizing spaces. Skipping item.")
                            continue
                    corrupted_book_for_test = original_book_text[:found_at] + actual_corrupted_sentence + original_book_text[found_at + len(actual_original_sentence):]
                except Exception as e: self.logger.error(f"    Error reconstructing for {target_tokens}, set {current_set_num}: {e}."); continue
                
                conceptual_original_book_id = f"original_document_s{target_tokens}_set{current_set_num}"
                conceptual_corrupted_book_id = f"corrupted_document_s{target_tokens}_set{current_set_num}_c{uuid.uuid4().hex[:4]}"
                
                if corruption_method == "delete_word": task_instr = "A single word was DELETED from one sentence in the Original Document to create the Modified Document. Identify the word that was DELETED."
                elif corruption_method == "replace_word": task_instr = "A single word in one sentence of the Original Document was REPLACED. Identify the ORIGINAL word that was replaced."
                else: continue
                ans_verify = orig_seg

                # ---- MODIFIED QUESTION TEXT TO INCLUDE FULL DOCUMENTS ----
                question_text = (
                    f"Instruction: You are given two long documents below, an 'Original Document' and a 'Modified Document'. "
                    f"The Modified Document was created by making a single word change (either a deletion or a replacement) in exactly one sentence of the Original Document. "
                    f"{task_instr}\n\n"
                    f"Respond by providing ONLY the single identified word, enclosed in <answer> tags. For example: <answer>the_word</answer>. "
                    f"Do not include any other text, reasoning, or explanations before or after the <answer> tags.\n\n"
                    f"--- START OF ORIGINAL DOCUMENT (ID: {conceptual_original_book_id}) ---\n"
                    f"{original_book_text}\n"
                    f"--- END OF ORIGINAL DOCUMENT ---\n\n"
                    f"--- START OF MODIFIED DOCUMENT (ID: {conceptual_corrupted_book_id}) ---\n"
                    f"{corrupted_book_for_test}\n"
                    f"--- END OF MODIFIED DOCUMENT ---"
                )
                # ---- END OF MODIFIED QUESTION TEXT ----

                q_id_suffix = f"s{target_tokens}_set{current_set_num}-fwNH-{uuid.uuid4().hex[:4]}" # NH for No Hint
                qa = QAItem(id=f"{self.name}-{q_id_suffix}", question=question_text, answer=str(ans_verify), modality='text', skill_coefficient=3,
                    metadata={'test_type': 'find_the_corrupted_word_full_context_no_hint', 
                              'corruption_type': corruption_method,
                              'book_target_tokens': target_tokens, 'book_actual_tokens_original': actual_book_tokens,
                              'book_set_num': current_set_num, 'book_version_id_for_cache': book_version_id,
                              # We no longer store the full texts in metadata if they are in the prompt itself
                              # to avoid extreme duplication if the QAItem object is logged/serialized.
                              # However, for debugging, you might want to keep them.
                              # 'original_book_text': original_book_text, 
                              # 'corrupted_book_text': corrupted_book_for_test,
                              'original_sentence_context_for_reference': actual_original_sentence, # Keep for our reference
                              'corrupted_sentence_context_for_reference': actual_corrupted_sentence, # Keep for our reference
                              'expected_original_segment': orig_seg, 'expected_corrupted_segment': corr_seg,
                              'expected_llm_output_format': "<answer>word</answer>"},
                    verification_fn=self._verify_corrupted_word_tagged)
                yield qa; total_items_yielded +=1
        self.logger.info(f"{self.name}: Finished. Total items yielded: {total_items_yielded} for {num_sets_to_generate} requested set(s).")

if __name__ == '__main__':
    print("Running LongContextWikiBookTest (Needle in Haystack - Full Docs in Prompt)...")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_run_lc_logs_full_prompt")
    os.makedirs(log_dir, exist_ok=True)
    
    # For this test, let's use a smaller token target to avoid overly long prompts during standalone run
    # but still demonstrate the principle.
    # The DEFAULT_TOKEN_TARGETS in the class is [1000, 10000, 100000]
    # Let's override it for the __main__ block for a quicker test.
    test_gen_config = {
        'data_dir': log_dir,
        'long_context_token_targets': [500, 1500] # Test with smaller "long" contexts for standalone
    } 
    lc_generator = LongContextWikiBookTest(config=test_gen_config)
    lc_generator.logger.setLevel(logging.DEBUG) 
    if hasattr(lc_generator, 'wiki_agent'): lc_generator.wiki_agent.logger.setLevel(logging.DEBUG)

    num_sets_to_request = 1 
    print(f"\n--- Requesting count={num_sets_to_request} ({num_sets_to_request} set(s) of up to {len(lc_generator.token_targets)} items each) ---")
    
    generated_items_list = []
    for i, item in enumerate(lc_generator.generate(count=num_sets_to_request, force_regenerate_books=True)):
        generated_items_list.append(item)
        print(f"\n--- Generated QAItem {i+1} (ID: {item.id}) ---")
        print(f"  TargetTokens: {item.metadata.get('book_target_tokens')}, Actual Original Tokens: {item.metadata.get('book_actual_tokens_original')}")
        print(f"  Expected Word: {item.answer}")
        print(f"  Question Snippet (first 300 chars):\n{item.question[:300]}...")
        print(f"  Question Snippet (last 200 chars):\n...{item.question[-200:]}")
        # To verify full documents are in metadata (if you choose to keep them there):
        # print(f"  Length of original_book_text in metadata: {len(item.metadata.get('original_book_text', ''))}")
        # print(f"  Length of corrupted_book_text in metadata: {len(item.metadata.get('corrupted_book_text', ''))}")


    if not generated_items_list: print(f"\nNo items generated. Check logs in '{log_dir}'.")
    else: print(f"\nSuccessfully generated {len(generated_items_list)} item(s).")
    
    print(f"\nTest logs (if any) might be in: {log_dir}")