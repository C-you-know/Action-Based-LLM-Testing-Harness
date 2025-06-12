# verify-auto.py
from knit_space import models, tests, Marker
import os
import tqdm
import time

os.environ["GEMINI_API_KEY"] = "type_your_api_key_here"

gemini = models.get_provider("google")

test_run_marker = Marker(
    action_base_flops=1.5e9,    
    action_flops_growth=0.30e9,   
    action_alpha=0.8,             
    action_model_name="gpt2"
)

test_cases = [
    tests.MathQATestClass,
    tests.RandomSentenceObfuscationTest,
    tests.CodingQATestClass,
    tests.LongContextWikiBookTest,
    tests.SudokuValidationQATest,
    tests.ChessMemoryQATest,
    tests.NRulesVectorQATest,
    tests.NRulesVectorFakeGuidanceQATest,
    tests.NthDecimalDigitQATest,
    tests.FindUniqueNumberIndexQATest,
    tests.WikiCharCountQATest,
    tests.MMLUObfuscatedQATest,
    tests.MMLUObfuscatedChoiceQATest,
    tests.MedMCQAObfuscatedChoiceQATest,
    tests.ImplicitCorrectionMathQATest
    ]

questions, answers = tests.create_test_cases(test_cases)

    
for i in tqdm.tqdm(range(len(questions))):

    query = questions[i]
    generated_answer = gemini.inference(
        model_name="models/gemini-2.0-flash",
        prompt=query.question,
        temperature=0.7,       
        max_output_tokens=12000)
    
    test_case_result = query.verify(generated_answer)
    test_run_marker.add_result(query, generated_answer, test_case_result)
    time.sleep(10)
    # print("Question" + query.question)
    # print("Answer" + str(query.answer))
    # print("GenAnswer" + generated_answer)
    # print("TestCaseResult" + str(test_case_result))


elo_score = test_run_marker.calculate_elo_score()
print(f"Final Elo Score: {elo_score:.6f}") 

test_run_marker.launch_review_server()