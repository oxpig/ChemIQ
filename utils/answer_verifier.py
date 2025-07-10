import os
import ast
import functools
import json
import signal

import requests
from rdkit import Chem

class TimeoutException(Exception):
    pass

def timeout(seconds=2):
    """
    Decorator to raise TimeoutException if the decorated function
    runs for more than `seconds` (Unix only, main thread).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def _handler(signum, frame):
                raise TimeoutException(f"Timed out after {seconds}s")
            # install alarm
            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            except TimeoutException:
                # fallback on timeout
                return {"is_correct": False, "opsin_smiles": None}
            finally:
                # cancel and restore
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator



class AnswerVerifier:

    def __init__(self, question_file):

        self._question_file = question_file
        self.all_questions = self._read_question_file(question_file)
        self.question_dict = {q["uuid"]: q for q in self.all_questions}


        # Cache for OPSIN requests
        os.makedirs('utils', exist_ok=True)
        self._opsin_cache_path = os.path.join('utils', 'opsin_cache.json')
        try:
            with open(self._opsin_cache_path, 'r') as f:
                self._opsin_cache = json.load(f)
        except (IOError, ValueError):
            # file missing or invalid JSON
            self._opsin_cache = {}
            

    def _read_question_file(self, question_file):
        all_questions = []
        with open(question_file, 'r') as f:
            for line in f:
                all_questions.append(json.loads(line))

        return all_questions
        
    #-------------- Methods for checking answers-------------------

    def _check_exact_match(self, model_answer, uuid):
        expected_answer = self.question_dict[uuid]["answer"]
        # Exact match of two strings
        return str(model_answer) == str(expected_answer)

    def _check_list_of_tuples(self, model_answer, uuid):
        """
        Verifies that model_answer matches the expected list of tuples.
        
        The expected answer can be provided as a string or as a list/tuple.
        Each element is normalized to a tuple if it is a list.
        """
        expected_answer = self.question_dict[uuid]["answer"]
        # Helper function to normalize an item to a tuple if possible.
        def normalize(item):
            return tuple(item) if isinstance(item, list) else item
    
        # Process the expected answer.
        if isinstance(expected_answer, str):
            try:
                expected_list = ast.literal_eval(expected_answer)
            except Exception as e:
                raise Exception(f"Error evaluating expected answer for uuid {uuid}: {e}")
        else:
            expected_list = expected_answer
    
        if not isinstance(expected_list, (list, tuple)):
            raise Exception(f"The expected answer for uuid {uuid} is not a list or tuple.")
    
        # Normalize each element in the expected answer.
        expected_normalized = set(normalize(item) for item in expected_list)

        try:
            model_answer = ast.literal_eval(model_answer)
        except Exception as e:
            model_answer = model_answer
                
        # Process and normalize the chat answer.
        if not isinstance(model_answer, (list, tuple)):
            print("Error in list_of_tuples verification: model_answer is not a list or tuple.")
            return False
        try:
            chat_normalized = set(normalize(item) for item in model_answer)
        except Exception as e:
            print("Error normalizing model_answer:", e)
            return False
    
        # Compare the two sets.
        return chat_normalized == expected_normalized

    
    def _check_range(self, model_answer, uuid):
        answer_range = self.question_dict[uuid]["answer_range"]
        try:
            value = float(model_answer)
            low, high = ast.literal_eval(answer_range)
            return low <= value <= high
        except Exception as e:
            print("Error in range verification:", e)
            return False

    def _canonicalize_smiles(self, smiles_str):
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return None
            Chem.RemoveStereochemistry(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None
        
    def _check_canonical_smi(self, model_answer, uuid):
        expected_answer = self.question_dict[uuid]["answer"]
        model_can = self._canonicalize_smiles(model_answer)
        expected_can = self._canonicalize_smiles(expected_answer)
        if model_can is None or expected_can is None:
            return False
        return model_can == expected_can
    
    def _get_opsin(self, model_answer):
        # Check cache
        if model_answer in self._opsin_cache:
            return self._opsin_cache[model_answer]

        # Do request
        base_url = "https://opsin.ch.cam.ac.uk/opsin/"
        api_url = f"{base_url}{model_answer}.json"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            smiles = data.get("smiles", "")
        except Exception as e:
            smiles = False

        self._opsin_cache[model_answer] = smiles
        try:
            with open(self._opsin_cache_path, 'w') as f:
                json.dump(self._opsin_cache, f, indent=2)
        except IOError:
            # unable to write cache—ignore silently
            pass

        return smiles


    @timeout(2)
    def check_answer(self, uuid, model_answer):
        is_correct = False
        opsin_smiles = None
        
        if uuid not in self.question_dict.keys():
            raise Exception(f"{uuid} is not contained in {self._question_file}")

        if not model_answer:
            # No model answer provided. Default fail.
            return {"is_correct": False, "opsin_smiles": None}

        if "\n" in str(model_answer) or len(str(model_answer)) > 200:
            return {"is_correct": False, "opsin_smiles": None}
        
        q = self.question_dict[uuid]
        method = q["verification_method"]
        
        answer_range = self.question_dict[uuid]["answer_range"]

        if method == "exact_match":
            is_correct = self._check_exact_match(model_answer, uuid)
            
        elif method == "list_of_tuples":
            is_correct = self._check_list_of_tuples(model_answer, uuid)
            
        elif method == "range":
            is_correct = self._check_range(model_answer, uuid)
            
        elif method == "canonical_smi_match":
            is_correct = self._check_canonical_smi(model_answer, uuid)
            
        elif method == "opsin":
            opsin_smiles = self._get_opsin(model_answer)
            is_correct = self._check_canonical_smi(opsin_smiles, uuid)
            
        else:
            raise ValueError(f"Unsupported verification method: {method}")
    
        return {"is_correct": is_correct, "opsin_smiles": opsin_smiles}

    def __repr__(self):
        from collections import Counter, defaultdict

        cat_counts = Counter(q["question_category"] for q in self.all_questions)
        subcat_counts = defaultdict(Counter)
        for q in self.all_questions:
            cat = q["question_category"]
            sub = q["sub_category"] or "<none>"
            subcat_counts[cat][sub] += 1

        parts = []
        for cat, total in cat_counts.items():
            subs = ", ".join(f"{s}={c}" for s, c in subcat_counts[cat].items())
            parts.append(f"{cat}={total}[{subs}]")

        summary = "\n ".join(parts)
        return f"Loaded {len(self.all_questions)} questions from {self._question_file}:\n {summary}"
