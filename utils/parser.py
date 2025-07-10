import ast
import json
import math
import re
import numpy as np

class AnswerParser:

    _BOXED_RE = re.compile(r'\\boxed\{\s*([^}]*?)\s*\}', re.DOTALL)
    _TEXT_RE = re.compile(r"\\text\{([^}]*)\}")
    _LAST_INT_RE = re.compile(r"([-+]?\d+)(?!.*\d)")
    _TUPLE_PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
    _LAST_FLOAT_RE = re.compile(r'([+-]?(?:\d*\.\d+|\d+\.\d*)(?:[eE][+-]?\d+)?)')

    def __init__(self, question_file, doClean=True):

        self._parsers = {
            "integer": self._parse_integer,
            "float": self._parse_float,
            "iupac": self._parse_iupac,
            "smiles": self._parse_smiles,
            "list_of_tuples": self._parse_list_of_tuples,

            # Legacy
            "string": self._parse_iupac,
        }

        self.all_questions = self._read_question_file(question_file)
        self.question_dict = {q["uuid"]: q for q in self.all_questions}
        self.doClean = doClean

        # Add rows for debug
        self.question_dict["integer"] = {"answer_format": "integer"}
        self.question_dict["float"] = {"answer_format": "float"}
        self.question_dict["iupac"] = {"answer_format": "iupac"}
        self.question_dict["smiles"] = {"answer_format": "smiles"}
        self.question_dict["list_of_tuples"] = {"answer_format": "list_of_tuples"}
        
        
    def parse(self, uuid, raw_answer):

        question = self.question_dict.get(uuid)
        if question is None:
            raise KeyError(f"Question UUID '{uuid}' not found in questions file")

        if isinstance(raw_answer, float) and math.isnan(raw_answer):
            return False

        answer_format = question["answer_format"]
        #method = question.get("verification_method")

        parser = self._parsers.get(answer_format)
        if parser is None:
            raise ValueError(f"Answer format '{answer_format}' not implemented")

        # do initial clean of strip and removing training "."
        raw_answer = raw_answer.strip().rstrip(".").strip()

        try:
            parsed = parser(str(raw_answer))
        except Exception:
            return False
        return parsed if parsed is not None else False

    def _read_question_file(self, question_file):
        all_questions = []
        with open(question_file, 'r') as f:
            for line in f:
                all_questions.append(json.loads(line))

        return all_questions

    #-------- Helpers --------------
    @staticmethod
    def _is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_list_literal(s: str) -> bool:
        try:
            return isinstance(ast.literal_eval(s), list)
        except (ValueError, SyntaxError):
            return False

    @classmethod
    def _last_line(cls, text: str) -> str:
        return next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")

    # ---------- Parser methods --------------
    def _parse_integer(self, answer):
        """
        Function for parsing an integer answer from text
        In order, method will check
        1. If answer only contains an integer and nothing else, return this integer
        2. Look for \boxed{} and return integer contained
        3. Check last line for an integer
        4. Check entire text and parse the final integer in all the text
        """

        # Most simple, when answer is only integer and nothing else
        if self._is_int(answer):
            return int(answer)

        if not self.doClean:
            return np.nan

        # Check if answer is written inside \boxed{}. This is typical of deepseek and other open source models
        boxed = self._BOXED_RE.findall(answer)
        if boxed:
            # Choose final boxed answer and check if integer
            if self._is_int(boxed[-1].strip()):
                return int(boxed[-1].strip())

        # Check if the last line in the string is an integer
        last_line = self._last_line(answer)
        if self._is_int(last_line.strip()):
            return int(last_line.strip())

        # Parse the final integer in the output text
        last_integer = self._LAST_INT_RE.findall(answer)
        if self._is_int(last_integer[-1]):
            return int(last_integer[-1])

        return None


    def _parse_float(self, answer):
        """
        Function for parsing an integer answer from text
        In order, method will check
        1. If answer only contains a float and nothing else, return this integer
        2. Look for \boxed{} and return integer contained
        3. Check last line for an integer
        4. Check entire text and parse the final integer in all the text
        """

        # Most simple, when answer is only float and nothing else
        if self._is_float(answer):
            return float(answer)

        if not self.doClean:
            return np.nan

        # Check if answer is written inside \boxed{}. This is typical of deepseek and other open source models
        boxed = self._BOXED_RE.findall(answer)
        if boxed:
            # Choose final boxed answer and check if float
            if self._is_float(boxed[-1].strip()):
                return float(boxed[-1].strip())

        # Check if the last line in the string is an integer
        last_line = self._last_line(answer)
        if self._is_float(last_line.strip()):
            return float(last_line.strip())

        # Parse the final float in the output text
        last_float = self._LAST_FLOAT_RE.findall(answer)
        if self._is_float(last_float[-1]):
            return float(last_float[-1])

        return None

    def _parse_iupac(self, answer):
        """
        Identified answer formats
        - String contained in \\boxed{}
        - String contained in \\text
        - Entire string contains IUPAC name only (will return last line only)

        - Length guard:
        - - Will not return answers with more than 300 characters 
        - - Will not return answers containing new lines within the text
        """
        def _valid(value):
            """
            Make sure the text isn't too long and doesn't contain new lines
            """
            value = value.strip()
            return value if 0 < len(value) <= 300 and "\n" not in value else None

        if not self.doClean:
            return answer
            
        # 1. \boxed{...}
        if (m := self._BOXED_RE.search(answer)):
            if v := _valid(m.group(1)):
                return v

        # 2. \text{...}
        if (m := self._TEXT_RE.search(answer)):
            if v := _valid(m.group(1)):
                return v

        # 3. Fallback – entire string (or last non‑empty line)
        return _valid(self._last_line(answer))


    def _parse_smiles(self, answer):
        """
        Identified answer formats
        - String contained in \\boxed{}
        - String contained in \\text
        - Entire string contains SMILES only
        - Sometimes answer is given as Answer: {smiles} - so will take the last answer after space
        """

        def _valid(value):
            """
            Make sure the text isn't too long and doesn't contain new lines
            """
            value = value.strip()
            return value if 0 < len(value) <= 300 and "\n" not in value else None

        if not self.doClean:
            return answer

        # 1. \boxed{...}
        if (m := self._BOXED_RE.search(answer)):
            if v := _valid(m.group(1)):
                return v

        # 2. \text{...}
        if (m := self._TEXT_RE.search(answer)):
            if v := _valid(m.group(1)):
                return v

        # 3. Fallback – entire string (or last non‑empty line)
        last_line = self._last_line(answer)
        last_word = last_line.split()[-1]
        return _valid(last_word)


    def _parse_list_of_tuples(self, answer):
        """
        Parsing list of tuples from answer
        This parser is the most complex. Many edge cases have been seen:
        - Ideal answer output looks like "[(0,0),(1,1),(1,1)]"
        - Sometimes the answer is printed on multiple lines "(0,0)\n(1,1)\n(1,1)"
        - Sometimes the answer has prime ' symbol to indicate different molecule "(0,0')\n(1,1')\n(1,1')"
        - Sometimes answer is list of lists [[0,0],[1,1],[2,2]]
        """
        # Remove prime ' symbols
        if self.doClean:
            answer = answer.replace("'", "")

        # If answer is already a list, return the list
        if self._is_list_literal(answer):
            answer_list = ast.literal_eval(answer)
            return [tuple(p) for p in answer_list]

        if not self.doClean:
            return None

        boxed_contents = self._BOXED_RE.findall(answer)
        if boxed_contents:
            content = boxed_contents[-1].strip()
            # Try literal eval on the boxed content
            if self._is_list_literal(content):
                answer_list = ast.literal_eval(content)
                return [tuple(p) for p in answer_list]
            # Otherwise, look for tuple pairs inside the box
            if pairs := self._TUPLE_PAIR_RE.findall(content):
                return [(int(x), int(y)) for x, y in pairs]

        # Search answer for tuples
        if pairs := self._TUPLE_PAIR_RE.findall(answer):
            return [(int(x), int(y)) for x, y in pairs]

        return None