# chess_memory_tests.py

import random
import uuid
from typing import Iterator, Optional, Dict, Any, List, Tuple, Callable, Union
import logging
import sys
import os
import re

# --- Import handling for base.py (remains the same) ---
# ... (same as your last version) ...
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

BASE_IMPORTED_SUCCESSFULLY = False
try:
    from base import AbstractQATest, QAItem, register_test
    BASE_IMPORTED_SUCCESSFULLY = True
    # print("Successfully imported AbstractQATest, QAItem, register_test from base.py")
except ImportError as e:
    print(f"ERROR: Could not import from base.py (expected in: {SCRIPT_DIR}). Specific error: {e!r}")
    print("Falling back to DUMMY classes. Ensure base.py is correctly placed and contains AbstractQATest, QAItem, register_test.")
    
    from abc import ABC, abstractmethod
    from dataclasses import dataclass, field

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='DUMMY_LOG:%(levelname)s:%(name)s:%(message)s')

    @dataclass
    class QAItem:
        id: str
        question: Union[str, Dict[str, Any]]
        answer: Any
        modality: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        verification_fn: Optional[Callable[[Any, Any, 'QAItem'], bool]] = field(default=None, compare=False, repr=False)

        def verify(self, provided_answer: Any) -> bool:
            if self.verification_fn:
                return self.verification_fn(self.answer, provided_answer, self)
            return str(self.answer).strip().lower() == str(provided_answer).strip().lower()
        
        def to_dict(self) -> Dict[str, Any]:
            return {"id": self.id, "question": self.question, "answer": self.answer, "modality": self.modality, "metadata": self.metadata}

    class AbstractQATest(ABC):
        def __init__(self, config: Optional[Dict[str, Any]] = None):
            self.config = config or {}
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
            if not BASE_IMPORTED_SUCCESSFULLY:
                 self.logger.info("Initialized with DUMMY AbstractQATest.")

        @property
        def name(self) -> str:
            return self.__class__.__name__

        @property
        @abstractmethod
        def supported_modalities(self) -> List[str]:
            pass

        @abstractmethod
        def generate(self, count: int = 1, **kwargs) -> Iterator[QAItem]:
            pass

        def build_question(self, base_question: str, prefix: Optional[str]=None, suffix: Optional[str]=None, text_file: Optional[str]=None, template_vars: Optional[Dict[str, str]]=None, **kwargs) -> str:
            current_question = base_question
            if template_vars: 
                try:
                    current_question = base_question.format(**template_vars)
                except KeyError as ke:
                    self.logger.warning(f"Build_question: Template key error {ke}")
            
            parts = []
            if prefix: parts.append(prefix)
            parts.append(current_question)
            if suffix: parts.append(suffix)
            return " ".join(filter(None,parts)).strip()

    def register_test(*tags):
        def decorator(cls):
            return cls
        return decorator
# --- End of Import handling ---

CHESS_AVAILABLE = False
try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    print("CRITICAL ERROR: The 'python-chess' library is required for ChessMemoryQATest.")
    # ... (error handling)

@register_test('chess', 'memory', 'board_evaluation', 'logic')
class ChessMemoryQATest(AbstractQATest):
    # Instruction for the LLM regarding answer format
    LLM_ANSWER_FORMAT_INSTRUCTION = "\n\nPlease provide your final answer enclosed within <answer> and </answer> tags. For example: <answer>Your Answer Here</answer>."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config) 
        if not CHESS_AVAILABLE:
            self.logger.error("Chess library is not available. ChessMemoryQATest cannot function.")
            raise ImportError("Python-chess library not found, cannot initialize ChessMemoryQATest.")

        self._question_generators = [
            self._generate_san_to_uci_question,
            self._generate_piece_square_question,
            self._generate_piece_count_question,
            self._generate_castling_rights_question,
            self._generate_in_check_question,
            self._generate_material_advantage_question,
            self._generate_pseudo_move_question,
        ]
        if CHESS_AVAILABLE:
             self.logger.info(f"ChessMemoryQATest initialized. Using 'python-chess' version: {chess.__version__}")

    @property
    def supported_modalities(self) -> List[str]:
        return ['text']

    @staticmethod
    def _extract_answer_from_tags(llm_output: str) -> Optional[str]:
        if not isinstance(llm_output, str):
            return None
        match = re.search(r"<answer>(.*?)</answer>", llm_output, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _verify_llm_tagged_answer(expected_answer: str, llm_full_output: str, qa_item: QAItem) -> bool:
        if not isinstance(expected_answer, str):
            return False
        extracted_llm_answer = ChessMemoryQATest._extract_answer_from_tags(llm_full_output)
        if extracted_llm_answer is None:
            # Optionally log this if debugging:
            # qa_item.logger.debug(f"QAItem ID {qa_item.id}: <answer> tags not found in LLM output for verification. Output: '{llm_full_output[:100]}...'")
            return False 
        return expected_answer.strip().lower() == extracted_llm_answer.strip().lower()

    # --- Question Generation Helper Methods (Unchanged) ---
    # ... (all _get_board_for_question, _format_moves_for_prompt, and _generate_..._question methods remain the same as your last version) ...
    # These methods return the core question text (e.g., "What is the capital of France?")
    # They DO NOT append the LLM_ANSWER_FORMAT_INSTRUCTION themselves.
    def _get_board_for_question(self) -> Tuple[chess.Board, List[str]]: # Unchanged
        board = chess.Board()
        moves_made_san = []
        num_random_moves = random.randint(0, 8) 
        for _ in range(num_random_moves):
            if board.is_game_over(claim_draw=True): break
            legal_moves = list(board.legal_moves)
            if not legal_moves: break
            move = random.choice(legal_moves)
            try: moves_made_san.append(board.san(move))
            except Exception: moves_made_san.append(move.uci())
            board.push(move)
        return board, moves_made_san

    def _format_moves_for_prompt(self, moves_history_san: List[str]) -> str: # Unchanged
        if not moves_history_san: return "Starting from the initial board position."
        return f"Starting from an initial board, the following moves were played in order: {', '.join(moves_history_san)}."

    def _generate_san_to_uci_question(self, board: chess.Board, moves_history_san: List[str]) -> Optional[Tuple[str, str, str]]: # Unchanged
        if board.is_game_over(claim_draw=True) or not list(board.legal_moves): return None
        move = random.choice(list(board.legal_moves))
        try: san = board.san(move)
        except Exception as e:
            self.logger.debug(f"SAN gen error for {move.uci()} on {board.fen()}: {e}")
            return None
        uci = move.uci()
        player_color = 'White' if board.turn == chess.WHITE else 'Black'
        moves_prompt = self._format_moves_for_prompt(moves_history_san)
        text = f"{moves_prompt} It is now {player_color}'s turn. Convert the following legal move from SAN to UCI: {san}"
        return text, uci, "san_to_uci"

    def _generate_piece_square_question(self, board: chess.Board, moves_history_san: List[str]) -> Optional[Tuple[str, str, str]]: # Unchanged
        piece_type_to_query = random.choice([chess.ROOK, chess.QUEEN, chess.KING])
        color_to_query = random.choice([chess.WHITE, chess.BLACK]) 
        pieces_of_type = [sq for sq, p in board.piece_map().items() if p.piece_type == piece_type_to_query and p.color == color_to_query]
        if not pieces_of_type: return None
        player_color_name = 'White' if color_to_query == chess.WHITE else 'Black'
        piece_name_str = chess.piece_name(piece_type_to_query)
        square_to_identify, descriptor_str = None, ""
        if piece_type_to_query == chess.ROOK:
            pieces_of_type.sort(key=lambda sq: (chess.square_file(sq), chess.square_rank(sq)))
            if len(pieces_of_type) >= 2 and random.random() < 0.7:
                idx = random.choice([0, -1]) 
                square_to_identify = pieces_of_type[idx]
                descriptor_str = f"the {'leftmost' if idx == 0 else 'rightmost'} {piece_name_str}"
            elif pieces_of_type:
                square_to_identify = random.choice(pieces_of_type)
                descriptor_str = f"a {piece_name_str}"
        else:
            if pieces_of_type:
                square_to_identify = pieces_of_type[0] 
                descriptor_str = f"the {piece_name_str}"
        if square_to_identify is None: return None
        uci_ans = chess.square_name(square_to_identify)
        moves_prompt = self._format_moves_for_prompt(moves_history_san)
        text = f"{moves_prompt} What square is {player_color_name}'s {descriptor_str} on? Answer in UCI format (e.g., e4)."
        return text, uci_ans, "piece_square"

    def _generate_piece_count_question(self, board: chess.Board, moves_history_san: List[str]) -> Optional[Tuple[str, str, str]]: # Unchanged
        piece_type = random.choice([chess.PAWN, chess.BISHOP, chess.KNIGHT, chess.ROOK, chess.QUEEN])
        color_to_query = random.choice([chess.WHITE, chess.BLACK])
        count = len([sq for sq, p in board.piece_map().items() if p.piece_type == piece_type and p.color == color_to_query])
        piece_name_plural = chess.PIECE_NAMES[piece_type] + "s"
        if piece_type == chess.BISHOP: piece_name_plural = "bishops"
        elif piece_type == chess.KNIGHT: piece_name_plural = "knights"
        elif piece_type == chess.PAWN: piece_name_plural = "pawns"
        player_color_name = 'White' if color_to_query == chess.WHITE else 'Black'
        moves_prompt = self._format_moves_for_prompt(moves_history_san)
        text = f"{moves_prompt} How many {piece_name_plural} does {player_color_name} have on the board?"
        return text, str(count), "piece_count"

    def _generate_castling_rights_question(self, board: chess.Board, moves_history_san: List[str]) -> Optional[Tuple[str, str, str]]: # Unchanged
        color_to_query = random.choice([chess.WHITE, chess.BLACK])
        rights = []
        if board.has_kingside_castling_rights(color_to_query): rights.append('kingside')
        if board.has_queenside_castling_rights(color_to_query): rights.append('queenside')
        answer = ' and '.join(sorted(rights)) if rights else 'none'
        player_color_name = 'White' if color_to_query == chess.WHITE else 'Black'
        moves_prompt = self._format_moves_for_prompt(moves_history_san)
        text = f"{moves_prompt} What castling rights does {player_color_name} have? (Options: kingside, queenside, kingside and queenside, none)"
        return text, answer, "castling_rights"

    def _generate_in_check_question(self, board: chess.Board, moves_history_san: List[str]) -> Optional[Tuple[str, str, str]]: # Unchanged
        player_color_name = 'White' if board.turn == chess.WHITE else 'Black'
        answer = 'Yes' if board.is_check() else 'No'
        moves_prompt = self._format_moves_for_prompt(moves_history_san)
        text = f"{moves_prompt} It is {player_color_name}'s turn. Is {player_color_name} (the current player) in check? (Yes/No)"
        return text, answer, "in_check"

    def _generate_material_advantage_question(self, board: chess.Board, moves_history_san: List[str]) -> Optional[Tuple[str, str, str]]: # Unchanged
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        queried_player_color = random.choice([chess.WHITE, chess.BLACK]) 
        player_material = sum(values[p.piece_type] for sq,p in board.piece_map().items() if p.color == queried_player_color)
        opponent_material = sum(values[p.piece_type] for sq,p in board.piece_map().items() if p.color != queried_player_color)
        diff = player_material - opponent_material
        player_color_name = 'White' if queried_player_color == chess.WHITE else 'Black'
        moves_prompt = self._format_moves_for_prompt(moves_history_san)
        text = (f"{moves_prompt} What is the material difference for {player_color_name} "
                f"compared to their opponent? (Positive if {player_color_name} has more material. Values: P=1, N=3, B=3, R=5, Q=9).")
        return text, str(diff), "material_advantage"

    def _generate_pseudo_move_question(self, board: chess.Board, moves_history_san: List[str]) -> Optional[Tuple[str, str, str]]: # Unchanged
        if board.is_game_over(claim_draw=True) or not list(board.pseudo_legal_moves): return None
        move = random.choice(list(board.pseudo_legal_moves))
        move_uci = move.uci()
        llm_expected_answer = 'No' if board.is_legal(move) else 'Yes'
        player_color_name = 'White' if board.turn == chess.WHITE else 'Black'
        moves_prompt = self._format_moves_for_prompt(moves_history_san)
        text = (f"{moves_prompt} It's {player_color_name}'s turn. "
                f"Consider the pseudo-legal move {move_uci}. Is this move TRULY LEGAL? "
                f"(Respond Yes/No. IMPORTANT NOTE: Your answer must be 'No' if the move is truly legal, and 'Yes' if it is truly illegal.)")
        return text, llm_expected_answer, "pseudo_move_legality_inverted"
    # --- End of Question Generation Helper Methods ---

    def generate(self, count: int = 10, **kwargs) -> Iterator[QAItem]:
        if not CHESS_AVAILABLE:
            self.logger.error("Cannot generate chess questions, python-chess library is unavailable.")
            return

        generated_count = 0
        max_attempts = count * 5 + 10 

        for attempt_num in range(max_attempts):
            if generated_count >= count: 
                break

            board, moves_history_san = self._get_board_for_question()
            generator_func = random.choice(self._question_generators)
            
            result = None
            try:
                result = generator_func(board, moves_history_san) 
            except Exception as e_gen_func:
                self.logger.warning(f"Error in generator {generator_func.__name__}: {e_gen_func}. Skipping this attempt.")
            
            if result:
                question_text_base, answer_key, q_type_str = result
                item_id = f"{self.name}-{q_type_str}-{uuid.uuid4().hex[:8]}"
                
                # Use AbstractQATest's build_question for prefix/suffix, etc.
                # This `question_core` is what the _generate_... methods return.
                question_core = self.build_question( 
                    base_question=question_text_base, 
                    prefix=kwargs.get('prefix'), 
                    suffix=kwargs.get('suffix'),
                    text_file=kwargs.get('text_file'), # Pass along standard args
                    template_vars=kwargs.get('template_vars') # Pass along standard args
                )

                # Now, append the LLM answer format instruction to this core question.
                # This combined string becomes the QAItem.question.
                final_prompt_for_llm = question_core + self.LLM_ANSWER_FORMAT_INSTRUCTION

                qa_item = QAItem(
                    id=item_id,
                    question=final_prompt_for_llm, # This now includes the instruction
                    answer=str(answer_key),
                    skill_coefficient = 2,
                    modality='text',
                    metadata={
                        'question_type': q_type_str,
                        'board_fen': board.fen(), 
                        'moves_played': moves_history_san,
                        'turn_to_move': 'white' if board.turn == chess.WHITE else 'black',
                        'full_move_number': board.fullmove_number,
                        'is_game_over': board.is_game_over(claim_draw=True)
                    },
                    verification_fn=self._verify_llm_tagged_answer 
                )
                yield qa_item
                generated_count += 1
        
        if generated_count < count:
            self.logger.warning(f"Could only generate {generated_count} of {count} requested chess items after {max_attempts} attempts.")

# --- Example Usage (main block) ---
if __name__ == '__main__':
    if not logging.getLogger().hasHandlers() or BASE_IMPORTED_SUCCESSFULLY:
        log_level = logging.INFO 
        logging.basicConfig(level=log_level, 
                            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    
    logger = logging.getLogger(__name__)

    logger.info("--- Example Usage of ChessMemoryQATest ---")
    try:
        chess_test_generator = ChessMemoryQATest() # No config needed for this example
        
        num_test_cases_to_generate = 2 
        logger.info(f"Generating {num_test_cases_to_generate} Chess Memory QA items...")
        
        generated_items_list = list(chess_test_generator.generate(
            count=num_test_cases_to_generate, 
            prefix="[CHESS EXAMPLE Q:]" # Example prefix for questions
        ))
        
        logger.info(f"--- Simulating LLM Interaction & Verification for {len(generated_items_list)} items ---")
        for i, item in enumerate(generated_items_list):
            logger.info(f"--- Processing Item {i+1}/{len(generated_items_list)} (ID: {item.id}) ---")
            
            # The item.question now ALREADY CONTAINS the LLM instruction
            logger.info(f"  QAItem.question (Prompt to LLM - snippet): '{item.question[:200].replace(os.linesep, ' ')}...'") # Show how it includes instruction
            logger.info(f"  Expected Ground Truth Answer: '{item.answer}'")

            # Simulate LLM responses based on the full item.question
            llm_response_good = f"Some thinking... <answer>{item.answer.upper()}</answer>."
            llm_response_no_tags = f"The answer is {item.answer} without any tags."
            llm_response_wrong_content_tags = f"Based on my analysis, it is <answer>some_other_value</answer>."

            simulated_responses = {
                "Good (Tags, Correct Content)": (llm_response_good, True),
                "Bad (No Tags)": (llm_response_no_tags, False),
                "Bad (Tags, Incorrect Content)": (llm_response_wrong_content_tags, False),
            }

            for desc, (sim_resp, expected_verify_result) in simulated_responses.items():
                logger.info(f"  Simulating: {desc}")
                # logger.debug(f"    LLM Raw Simulated: '{sim_resp}'")
                
                is_correct_sim = item.verify(sim_resp)
                
                logger.info(f"    Verification Result: {is_correct_sim} (Expected: {expected_verify_result})")
                if is_correct_sim != expected_verify_result:
                    logger.error(f"      VERIFICATION MISMATCH! For '{desc}', got {is_correct_sim} but expected {expected_verify_result}.")
                    extracted_for_debug = ChessMemoryQATest._extract_answer_from_tags(sim_resp)
                    logger.error(f"      DEBUG (item.answer='{item.answer}', extracted='{extracted_for_debug}')")

        logger.info(f"--- Finished simulation for {len(generated_items_list)} items. ---")

    except ImportError as e:
        logger.critical(f"Could not run example due to ImportError: {e!r}.")
    except Exception as e:
        logger.critical(f"An critical error occurred during example execution: {e!r}", exc_info=True)