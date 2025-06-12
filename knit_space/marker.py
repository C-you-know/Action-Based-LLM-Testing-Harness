# knit_space/marker.py
from typing import List, Dict, Any, Tuple
import math
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import warnings

# Try to import Flask, make it an optional dependency
try:
    from flask import Flask, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    # Define dummy decorators if Flask is not available, so the rest of the code doesn't break
    class DummyFlask:
        def route(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
        def run(self, *args, **kwargs):
            pass
    # print("Warning: Flask is not installed. Web review server will not be available.") # Optional warning

warnings.filterwarnings("ignore", message="`return_dict_in_generate` is NOT set to `True`, but `output_hidden_states` is.", category=UserWarning)
warnings.filterwarnings("ignore", message="`loss_type=None` was set in the config but it is unrecognised.", category=UserWarning)

QAItemType = Any 

class Marker:
    _flask_app = None # Class variable to hold the Flask app instance
    _marker_instance_for_flask = None # Class variable to hold the marker instance for Flask routes

    def __init__(self, 
                 action_base_flops: float = 1.0e9, 
                 action_flops_growth: float = 0.25e9, 
                 action_alpha: float = 1.0,
                 action_model_name: str = "gpt2"):
        # ... (initialization of results, counts, action params, and action model/tokenizer is THE SAME)
        self.results: List[Tuple[QAItemType, str, bool]] = []
        self.attempted_count: int = 0
        self.correct_count: int = 0
        self.failed_count: int = 0

        self.action_base_flops = action_base_flops
        self.action_flops_growth = action_flops_growth
        self.action_alpha = action_alpha
        
        self.action_tokenizer = None
        self.action_model = None
        if FLASK_AVAILABLE: # Only attempt to load heavy models if Flask is there for review
            try:
                # print(f"Marker: Initializing action model ({action_model_name})...") # Optional
                self.action_tokenizer = GPT2Tokenizer.from_pretrained(action_model_name)
                if self.action_tokenizer.eos_token is None:
                    self.action_tokenizer.eos_token = self.action_tokenizer.pad_token if self.action_tokenizer.pad_token else "<|endoftext|>"
                self.action_model = GPT2LMHeadModel.from_pretrained(action_model_name, output_hidden_states=True)
                self.action_model.eval()
                # print(f"Marker: Action model ({action_model_name}) initialized successfully.") # Optional
            except Exception as e:
                print(f"ERROR: Could not load action model '{action_model_name}'. Action/Elo calculation will effectively be zero. Error: {e}")
        else:
            print("Marker: Flask not available, action model for trajectory S computation will not be loaded (as it's mainly for review details here).")


    # add_result, get_summary_statistics, _compute_action_trajectory_S_value, 
    # _calculate_item_elo_component, calculate_elo_score
    # --- ARE EXACTLY THE SAME as the previous version ---
    def add_result(self, qa_item: QAItemType, generated_answer: str, test_result: bool):
        if not isinstance(test_result, bool):
            raise ValueError(f"test_result for Marker.add_result must be a boolean. Got: {type(test_result)}")
        self.results.append((qa_item, generated_answer, test_result))
        self.attempted_count += 1
        if test_result:
            self.correct_count += 1
        else:
            self.failed_count += 1

    def get_summary_statistics(self) -> Dict[str, int]:
        return {
            "attempted": self.attempted_count,
            "correct": self.correct_count,
            "failed": self.failed_count,
        }

    def _compute_action_trajectory_S_value(self, prompt: str, answer: str) -> float:
        if not self.action_model or not self.action_tokenizer: # Check if models loaded
            # print("Warning: Action model/tokenizer not available for S-value. Returning inf.") # Optional
            return float('inf') 

        full_text = prompt + answer
        if not full_text.strip():
            return float('inf')

        try:
            inputs = self.action_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.action_tokenizer.model_max_length)
            input_ids = inputs['input_ids'][0]
            seq_len = input_ids.shape[0]

            if seq_len <= 1: 
                return float('inf') 

            with torch.no_grad():
                outputs = self.action_model(**inputs, labels=inputs['input_ids'])

            hidden_states = outputs.hidden_states[-1].squeeze(0)
            logits = outputs.logits.squeeze(0)
            
            if hidden_states.nelement() == 0 or logits.nelement() == 0 or hidden_states.shape[0] != seq_len or logits.shape[0] != seq_len :
                return float('inf')

            log_probs_per_token = []
            for t in range(seq_len - 1):
                log_probs_dist = F.log_softmax(logits[t], dim=-1)
                actual_next_token_id = input_ids[t + 1]
                log_prob_of_token = log_probs_dist[actual_next_token_id].item()
                log_probs_per_token.append(log_prob_of_token)

            L = []
            for t in range(seq_len - 1):
                flops = self.action_base_flops + self.action_flops_growth * t
                v_norm_sq_tensor = torch.norm(hidden_states[t + 1]) 
                v_norm_sq = (v_norm_sq_tensor ** 2).item()
                
                kinetic = 0.5 * flops * v_norm_sq
                potential = -log_probs_per_token[t]
                lagrangian_t = kinetic + self.action_alpha * potential
                L.append(lagrangian_t)
            
            S = sum(L)
            if not math.isfinite(S):
                return float('inf')
            return S
        except Exception as e:
            # print(f"Error in _compute_action_trajectory_S_value: {e}") # Optional
            return float('inf')

    def _calculate_item_elo_component(self, qa_item: QAItemType, generated_answer: str, result_bool: bool) -> float:
        if not result_bool:
            return 0.0

        prompt_text = str(getattr(qa_item, 'question', ''))
        S_value = self._compute_action_trajectory_S_value(prompt_text, generated_answer)

        if S_value <= 0 or not math.isfinite(S_value) or S_value == float('inf'):
            return 0.0 
        
        cost = math.log10(S_value)

        if cost == 0 or not math.isfinite(cost):
            return 1.0 / 1e9 # Effectively zero for S=1 or problematic cost
        
        return 1.0 / cost

    def calculate_elo_score(self) -> float:
        if not self.results:
            return 0.0 

        total_elo_score = 0.0
        for qa_item, generated_answer, result_bool in self.results:
            try:
                skill_coeff = float(getattr(qa_item, 'skill_coefficient', 1.0))
            except (ValueError, TypeError):
                skill_coeff = 1.0
            
            item_elo_component = self._calculate_item_elo_component(
                qa_item, generated_answer, result_bool
            )
            total_elo_score += item_elo_component * skill_coeff
        return total_elo_score
    
    # --- NEW/UPDATED Method for Phase 4 ---
    def launch_review_server(self, host="localhost", port=8000):
        if not FLASK_AVAILABLE:
            print("Flask is not installed. Cannot launch web review server.")
            print("Falling back to console summary:")
            # Fallback to console print if Flask is not available
            if not self.results:
                print("  Marker: No results to display.")
                return
            print("\n--- Review Data (Console Fallback) ---")
            for i, (q_item, gen_ans, res_bool) in enumerate(self.results):
                # ... (console print logic from previous version) ...
                q_id = getattr(q_item, 'id', 'N/A')
                q_skill_coeff = getattr(q_item, 'skill_coefficient', 'N/A')
                prompt_text = str(getattr(q_item, 'question', ''))
                s_val_item = self._compute_action_trajectory_S_value(prompt_text, gen_ans)
                cost_item = -float('inf') 
                inv_action_item = 0.0
                if res_bool and s_val_item > 0 and math.isfinite(s_val_item):
                    cost_item = math.log10(s_val_item)
                    if cost_item != 0 and math.isfinite(cost_item):
                        inv_action_item = 1.0 / cost_item
                print(f"\nItem {i+1}: {'PASS' if res_bool else 'FAIL'} (verify())")
                print(f"  ID: {q_id}, Skill Coeff: {q_skill_coeff}")
                if res_bool: print(f"  Trajectory S: {s_val_item:.4e}, Cost (log10S): {cost_item:.4f}, InvAction (1/Cost): {inv_action_item:.6f}")
                else: print(f"  (Failed verify, InvAction: 0.0)")
            print("--- End of Review ---")
            return

        # Use class variables to avoid re-creating app and to pass data to routes
        if Marker._flask_app is None:
            Marker._flask_app = Flask(__name__)

            @Marker._flask_app.route('/')
            def index():
                # Access the marker instance that called launch_review_server
                marker_instance = Marker._marker_instance_for_flask 
                if not marker_instance:
                    return "Error: Marker instance not found for Flask app."

                # Prepare data for the template
                display_results = []
                for q_item, gen_ans, res_bool in marker_instance.results:
                    s_val = marker_instance._compute_action_trajectory_S_value(str(getattr(q_item, 'question', '')), gen_ans)
                    cost = 0.0
                    inv_action = 0.0
                    if res_bool and s_val > 0 and math.isfinite(s_val):
                        cost = math.log10(s_val)
                        if cost != 0 and math.isfinite(cost):
                            inv_action = 1.0 / cost
                    
                    display_results.append({
                        "id": getattr(q_item, 'id', 'N/A'),
                        "question_text": str(getattr(q_item, 'question', 'N/A')),
                        "expected_info": str(getattr(q_item, 'answer', 'N/A')),
                        "generated_answer_text": gen_ans,
                        "passed_verify": res_bool,
                        "skill_coeff": getattr(q_item, 'skill_coefficient', 1.0),
                        "s_value": f"{s_val:.4e}",
                        "cost_log10s": f"{cost:.4f}" if math.isfinite(cost) else "N/A (bad S)",
                        "inv_action": f"{inv_action:.6f}"
                    })
                
                # Calculate Elo score again for display, or retrieve if stored
                current_elo_score = marker_instance.calculate_elo_score()


                html_template = """
                <!DOCTYPE html>
                <html><head><title>KnitSpace Test Run Review</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
                    h1, h2 { color: #333; }
                    .summary { background-color: #fff; padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .result-item { background-color: #fff; border: 1px solid #ddd; margin-bottom: 20px; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .fail { border-left: 5px solid #d9534f; } 
                    .pass { border-left: 5px solid #5cb85c; }
                    pre { white-space: pre-wrap; word-wrap: break-word; background-color: #e9e9e9; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
                    table { width: 100%; border-collapse: collapse; margin-top:10px; }
                    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
                    th { background-color: #5cb85c; color: white; }
                    .fail th { background-color: #d9534f; }
                </style></head><body>
                <h1>KnitSpace Test Run Review</h1>
                <div class="summary">
                    <h2>Summary Statistics</h2>
                    <p>Total Attempted: {{ stats.attempted }}</p>
                    <p>Correctly Solved: {{ stats.correct }}</p>
                    <p>Failed: {{ stats.failed }}</p>
                    <p><strong>Final Elo Score:</strong> {{ "%.6f" | format(elo_score) }}</p>
                </div>

                {% for item in results_data %}
                    <div class="result-item {% if item.passed_verify %}pass{% else %}fail{% endif %}">
                        <table>
                            <tr class="{% if item.passed_verify %}pass{% else %}fail{% endif %}">
                                <th colspan="2">Item {{ loop.index }} (ID: {{ item.id }}) - {% if item.passed_verify %}PASS{% else %}FAIL{% endif %} (verify())</th>
                            </tr>
                            <tr><td>Skill Coefficient</td><td>{{ item.skill_coeff }}</td></tr>
                            {% if item.passed_verify %}
                                <tr><td>Trajectory S</td><td>{{ item.s_value }}</td></tr>
                                <tr><td>Cost (log10S)</td><td>{{ item.cost_log10s }}</td></tr>
                                <tr><td>InvAction (1/Cost)</td><td>{{ item.inv_action }}</td></tr>
                            {% else %}
                                <tr><td>InvAction (1/Cost)</td><td>0.000000 (Failed verify)</td></tr>
                            {% endif %}
                        </table>
                        <h3>Question:</h3><pre>{{ item.question_text }}</pre>
                        <h3>Expected Info (qa_item.answer):</h3><pre>{{ item.expected_info }}</pre>
                        <h3>Generated Answer:</h3><pre>{{ item.generated_answer_text }}</pre>
                    </div>
                {% endfor %}
                </body></html>
                """
                return render_template_string(html_template, 
                                              results_data=display_results, 
                                              stats=marker_instance.get_summary_statistics(),
                                              elo_score=current_elo_score)

        Marker._marker_instance_for_flask = self # Store current instance for the route
        print(f"Starting review server on http://{host}:{port}")
        print("Press CTRL+C to quit the web server.")
        try:
            Marker._flask_app.run(host=host, port=port, debug=False) # debug=False is safer
        except Exception as e_flask:
            print(f"Failed to start Flask server: {e_flask}")
            print("Ensure Flask is installed: pip install Flask")