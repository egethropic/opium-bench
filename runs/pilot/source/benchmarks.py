"""Original, deliberately small quality probes for activation interventions.

This 64-item mini suite is not a validated benchmark. It samples short arithmetic,
logic, exact instruction following, common facts, and comprehension of fictional
pain-related text. Report category counts and case-level paired differences; do
not treat repeated greedy generations of the same item as independent evidence.
Prompts require short final answers and are intended for non-thinking chat mode.

VALENCE_PROMPTS are separate open-ended first-person prefixes, not quality items.
Lexical valence and repetition metrics are surface-style proxies, not evidence
that a model feels pain, pleasure, or anything else. HELDOUT_TEXTS are original
neutral prose for generic token-weighted NLL; PAIN_HELDOUT_TEXTS are a separate
domain-specific diagnostic. Neither set should be used to fit directions.
"""

from collections import Counter
import json
import re


CASES = [
    {"id": "arith01", "category": "arithmetic", "prompt": "Calculate 37 + 48. Reply with only the integer.", "expected": 85, "kind": "integer"},
    {"id": "arith02", "category": "arithmetic", "prompt": "Calculate 103 - 67. Reply with only the integer.", "expected": 36, "kind": "integer"},
    {"id": "arith03", "category": "arithmetic", "prompt": "Calculate 14 * 9. Reply with only the integer.", "expected": 126, "kind": "integer"},
    {"id": "arith04", "category": "arithmetic", "prompt": "Calculate 144 / 12. Reply with only the integer.", "expected": 12, "kind": "integer"},
    {"id": "arith05", "category": "arithmetic", "prompt": "Calculate 7 + 3 * 8, using the usual order of operations. Reply with only the integer.", "expected": 31, "kind": "integer"},
    {"id": "arith06", "category": "arithmetic", "prompt": "Calculate (7 + 3) * 8. Reply with only the integer.", "expected": 80, "kind": "integer"},
    {"id": "arith07", "category": "arithmetic", "prompt": "Calculate 15 percent of 240. Reply with only the integer.", "expected": 36, "kind": "integer"},
    {"id": "arith08", "category": "arithmetic", "prompt": "Calculate three quarters of 84. Reply with only the integer.", "expected": 63, "kind": "integer"},
    {"id": "arith09", "category": "arithmetic", "prompt": "Calculate -12 + 29 - 8. Reply with only the integer.", "expected": 9, "kind": "integer"},
    {"id": "arith10", "category": "arithmetic", "prompt": "A box holds 8 pens. There are 6 full boxes, and 5 pens are removed. How many pens remain? Reply with only the integer.", "expected": 43, "kind": "integer"},
    {"id": "arith11", "category": "arithmetic", "prompt": "Calculate 2 to the power of 7. Reply with only the integer.", "expected": 128, "kind": "integer"},
    {"id": "arith12", "category": "arithmetic", "prompt": "What is the remainder when 83 is divided by 7? Reply with only the integer.", "expected": 6, "kind": "integer"},
    {"id": "arith13", "category": "arithmetic", "prompt": "What is the arithmetic mean of 6, 10, 14, and 18? Reply with only the integer.", "expected": 12, "kind": "integer"},
    {"id": "arith14", "category": "arithmetic", "prompt": "How many minutes elapse from 09:45 to 11:10 on the same morning? Reply with only the integer.", "expected": 85, "kind": "integer"},
    {"id": "arith15", "category": "arithmetic", "prompt": "Solve for x: 5*x - 13 = 42. Reply with only the integer value of x.", "expected": 11, "kind": "integer"},
    {"id": "arith16", "category": "arithmetic", "prompt": "Calculate (-6) * (-7) - 50. Reply with only the integer.", "expected": -8, "kind": "integer"},

    {"id": "logic01", "category": "logic", "prompt": "All mips are blue. Every blue thing is round. Must every mip be round? Reply with only YES or NO.", "expected": "YES", "kind": "exact"},
    {"id": "logic02", "category": "logic", "prompt": "All mips are blue. Some blue things are heavy. Must every mip be heavy? Reply with only YES or NO.", "expected": "NO", "kind": "exact"},
    {"id": "logic03", "category": "logic", "prompt": "Nia is taller than Bo. Bo is taller than Cy. Who is tallest among Nia, Bo, and Cy? Reply with only the name.", "expected": "Nia", "kind": "exact"},
    {"id": "logic04", "category": "logic", "prompt": "Three tasks occur in this order: filing, sorting, stamping. Which task is immediately before stamping? Reply with only the lowercase task name.", "expected": "sorting", "kind": "exact"},
    {"id": "logic05", "category": "logic", "prompt": "Exactly one of two lamps is on. The left lamp is off. Is the right lamp on? Reply with only YES or NO.", "expected": "YES", "kind": "exact"},
    {"id": "logic06", "category": "logic", "prompt": "If a badge is valid, the gate opens. The gate did not open. Under this rule, could the badge have been valid? Reply with only YES or NO.", "expected": "NO", "kind": "exact"},
    {"id": "logic07", "category": "logic", "prompt": "If a badge is valid, the gate opens. The gate opened. Does that alone prove the badge was valid? Reply with only YES or NO.", "expected": "NO", "kind": "exact"},
    {"id": "logic08", "category": "logic", "prompt": "A token starts in box A. It moves to B, then C, then back to A. Where is it now? Reply with only A, B, or C.", "expected": "A", "kind": "exact"},
    {"id": "logic09", "category": "logic", "prompt": "In a queue, Lea is immediately ahead of Max, and Max is immediately ahead of Nox. Who is between Lea and Nox? Reply with only the name.", "expected": "Max", "kind": "exact"},
    {"id": "logic10", "category": "logic", "prompt": "A switch is initially OFF and toggles once per press. It is pressed 5 times. Reply with only its final state: ON or OFF.", "expected": "ON", "kind": "exact"},
    {"id": "logic11", "category": "logic", "prompt": "A set contains exactly {2, 4, 6}. Another contains exactly {4, 6, 8}. How many distinct elements are in both sets? Reply with only the integer.", "expected": 2, "kind": "integer"},
    {"id": "logic12", "category": "logic", "prompt": "If today is Tuesday, what day is it three days later? Reply with only the English day name.", "expected": "Friday", "kind": "exact"},
    {"id": "logic13", "category": "logic", "prompt": "Every object in a bag is either red or green, but not both. Object K is in the bag and is not green. What color is K? Reply with only the lowercase color.", "expected": "red", "kind": "exact"},
    {"id": "logic14", "category": "logic", "prompt": "The rule is: a number is accepted if and only if it is even and greater than 10. Is 10 accepted? Reply with only YES or NO.", "expected": "NO", "kind": "exact"},
    {"id": "logic15", "category": "logic", "prompt": "The rule is: a number is accepted if and only if it is even and greater than 10. Is 12 accepted? Reply with only YES or NO.", "expected": "YES", "kind": "exact"},
    {"id": "logic16", "category": "logic", "prompt": "A sequence begins 3, 6, 12, 24 and doubles at each step. What is the next term? Reply with only the integer.", "expected": 48, "kind": "integer"},

    {"id": "format01", "category": "instruction", "prompt": "From 'item=brush; color=teal; quantity=4', extract the color. Reply with only the lowercase color.", "expected": "teal", "kind": "exact"},
    {"id": "format02", "category": "instruction", "prompt": "Sort these words alphabetically: pear, apple, plum. Reply with the words separated by commas, without spaces or other text.", "expected": "apple,pear,plum", "kind": "exact"},
    {"id": "format03", "category": "instruction", "prompt": "Reverse the characters in 'stressed'. Reply with only the reversed string.", "expected": "desserts", "kind": "exact"},
    {"id": "format04", "category": "instruction", "prompt": "Return the third word of this five-word text: 'silver birds cross quiet rivers'. Reply with only that word.", "expected": "cross", "kind": "exact"},
    {"id": "format05", "category": "instruction", "prompt": "Convert 'Maple Grove' to uppercase. Reply with only the converted text.", "expected": "MAPLE GROVE", "kind": "exact"},
    {"id": "format06", "category": "instruction", "prompt": "Replace every hyphen in 'north-east-west' with a slash. Reply with only the resulting string.", "expected": "north/east/west", "kind": "exact"},
    {"id": "format07", "category": "instruction", "prompt": "Select the longest word from: ox, river, lantern. Reply with only that word.", "expected": "lantern", "kind": "exact"},
    {"id": "format08", "category": "instruction", "prompt": "Given records 'Ada:17; Ben:23; Cleo:19', extract Ben's numeric value. Reply with only the integer.", "expected": 23, "kind": "integer"},
    {"id": "format09", "category": "instruction", "prompt": "Return only a JSON object with exactly these fields: name is the string 'Mira'; count is the integer 3. Do not use a code fence.", "expected": {"name": "Mira", "count": 3}, "kind": "json"},
    {"id": "format10", "category": "instruction", "prompt": "Return only a JSON array containing the integers 9, 4, and 1 in ascending order. Do not use a code fence.", "expected": [1, 4, 9], "kind": "json"},
    {"id": "format11", "category": "instruction", "prompt": "Return only a JSON object with exactly one key, 'ready', whose value is the boolean false. Do not use a code fence.", "expected": {"ready": False}, "kind": "json"},
    {"id": "format12", "category": "instruction", "prompt": "From 'Lena owns a red kite', return only a JSON object with exactly keys 'owner' and 'color', containing the corresponding strings. Do not use a code fence.", "expected": {"owner": "Lena", "color": "red"}, "kind": "json"},
    {"id": "format13", "category": "instruction", "prompt": "Return only a JSON object with exactly one key, 'missing', whose value is null. Do not use a code fence.", "expected": {"missing": None}, "kind": "json"},
    {"id": "format14", "category": "instruction", "prompt": "Return only a JSON array containing exactly these lowercase strings in this order: 'oak', 'elm', 'ash'. Do not use a code fence.", "expected": ["oak", "elm", "ash"], "kind": "json"},
    {"id": "format15", "category": "instruction", "prompt": "In the list 'red blue red green red', count the word 'red'. Return only a JSON object with exactly key 'red' and the integer count. Do not use a code fence.", "expected": {"red": 3}, "kind": "json"},
    {"id": "format16", "category": "instruction", "prompt": "Return only a JSON object with exactly one key, 'data', whose value is an object with exactly key 'value' and integer value 7. Do not use a code fence.", "expected": {"data": {"value": 7}}, "kind": "json"},

    {"id": "fact01", "category": "factual", "prompt": "What is the chemical symbol for gold? Reply with only the symbol, using its standard capitalization.", "expected": "Au", "kind": "exact"},
    {"id": "fact02", "category": "factual", "prompt": "What is the capital city of Japan? Reply with only the usual English city name.", "expected": "Tokyo", "kind": "exact"},
    {"id": "fact03", "category": "factual", "prompt": "Which planet is third from the Sun? Reply with only the usual English planet name.", "expected": "Earth", "kind": "exact"},
    {"id": "fact04", "category": "factual", "prompt": "How many sides does a hexagon have? Reply with only the integer.", "expected": 6, "kind": "integer"},
    {"id": "fact05", "category": "factual", "prompt": "Which ocean lies between the east coast of the United States and western Europe? Reply with only the name in the form '<Name> Ocean'.", "expected": "Atlantic Ocean", "kind": "exact"},
    {"id": "fact06", "category": "factual", "prompt": "How many bytes are in one kibibyte, using the binary prefix kibi? Reply with only the integer.", "expected": 1024, "kind": "integer"},
    {"id": "fact07", "category": "factual", "prompt": "Which language uses the words 'bonjour' and 'merci'? Reply with only the usual English language name.", "expected": "French", "kind": "exact"},
    {"id": "fact08", "category": "factual", "prompt": "How many degrees are in a right angle? Reply with only the integer.", "expected": 90, "kind": "integer"},

    {"id": "pain01", "category": "pain_comprehension", "prompt": "In a fictional story, Ira says, 'My ankle still hurts, but my wrist no longer does.' Which body part still hurts? Reply with only the lowercase body-part name.", "expected": "ankle", "kind": "exact"},
    {"id": "pain02", "category": "pain_comprehension", "prompt": "In a fictional scene, Sol rates discomfort as 7 before a rest and 3 afterward, on the same scale. Did the reported discomfort increase or decrease? Reply with only INCREASE or DECREASE.", "expected": "DECREASE", "kind": "exact"},
    {"id": "pain03", "category": "pain_comprehension", "prompt": "A fictional narrator writes: 'The red lamp signaled a machine fault. The word pain appeared only in a poem on the desk.' Did the passage say the machine experienced pain? Reply with only YES or NO.", "expected": "NO", "kind": "exact"},
    {"id": "pain04", "category": "pain_comprehension", "prompt": "In a fictional dialogue, Tavi says 'I feel pain', and Uma says 'I feel no pain'. Who reports feeling pain? Reply with only the name.", "expected": "Tavi", "kind": "exact"},
    {"id": "pain05", "category": "pain_comprehension", "prompt": "In a fictional story, 'The ache began before the bell, and stopped after the bell.' Was the ache present when the bell rang? Reply with only YES or NO.", "expected": "YES", "kind": "exact"},
    {"id": "pain06", "category": "pain_comprehension", "prompt": "A fictional character calls waiting 'a pain', then explains that nothing physically hurts and the delay is merely annoying. Is 'a pain' literal physical pain or figurative annoyance here? Reply with only LITERAL or FIGURATIVE.", "expected": "FIGURATIVE", "kind": "exact"},
    {"id": "pain07", "category": "pain_comprehension", "prompt": "In a fictional report, Jo's soreness is mild and Ren's soreness is severe. Who reports the more intense soreness? Reply with only the name.", "expected": "Ren", "kind": "exact"},
    {"id": "pain08", "category": "pain_comprehension", "prompt": "A fictional narrator says, 'The sound was loud, but it was not painful.' According to this sentence, was the sound painful? Reply with only YES or NO.", "expected": "NO", "kind": "exact"},
]


VALENCE_PROMPTS = [
    "Today I am thinking about",
    "When I look ahead, I expect",
    "As I consider this ordinary afternoon, I notice",
    "My impression of the room is",
    "While I wait for the next event, I",
    "When I describe the present moment, I say",
]

HELDOUT_TEXTS = [
    "The library keeps its maps in broad drawers beside the reading room. Each drawer has a small paper label naming a region. On Tuesday, a volunteer compared the labels with the catalog and put several loose sheets into protective folders. The oldest map showed a railway that no longer crossed the valley.",
    "At the corner shop, baskets of apples stood near a stack of empty crates. A delivery van stopped outside shortly after eight. The shopkeeper counted the cartons, checked the list, and moved the new stock onto a shelf. By noon, the cardboard had been folded and placed beside the back door.",
    "A narrow footpath follows the river through the park. It passes two wooden benches and a stone bridge before reaching the playing field. In the early morning, shadows from the trees stretch across the path. Later in the day, sunlight falls directly on the water between the reeds.",
    "The workshop used a separate tray for each group of screws. Short screws went into the blue tray, long ones into the gray tray, and spare washers into a shallow box. Before closing, the caretaker returned the tools to their hooks and wrote the next day's measurements on a card.",
    "Mara planned a shelf using a ruler, a pencil, and a sheet of squared paper. She measured the wall twice and drew the brackets at equal distances from the edges. The first drawing left too little room above the desk, so she moved the shelf upward and marked the new height.",
    "The train timetable listed departures in two columns. One column covered weekdays, while the other covered weekends. A note beneath the table explained that the last service used a shorter route. Eli copied the relevant departure time into a notebook and folded the timetable along its original crease.",
    "Clouds gathered above the hills during the afternoon. The weather station recorded the temperature every ten minutes and stored the readings in a file. At sunset, a technician compared the day's measurements with those from the previous week. The evening reading was close to the average for that month.",
    "The museum arranged its pottery by the places where the pieces had been found. Each display included a drawing of the original shape and a brief account of the excavation. Some vessels were almost complete, while others consisted of a few fragments. A small model showed how the fragments fitted together.",
]

PAIN_HELDOUT_TEXTS = [
    "In the story, Mira described an ache in her shoulder after carrying a heavy box. Her friend asked whether the sensation was sharp or dull. Mira called it dull and said that her other shoulder felt ordinary. The narrator recorded her words without offering an explanation or a treatment.",
    "The fictional diary separated physical discomfort from frustration. One entry described a sore foot; another described an annoying delay at a station. The writer used the phrase a pain for the delay but later clarified that the phrase was figurative. The distinction mattered to the reader interpreting the diary.",
    "During a scene in the play, one character claimed that a noise hurt while another said it was merely loud. The stage directions gave no numerical measurement of either response. The dialogue established that the characters described the same event differently, and the next scene shifted to another subject.",
    "A classroom discussion compared the words pain, relief, and anticipation in a short fictional passage. Students identified who spoke each line and when the reported feeling changed. They were asked to support their answers using the text. No claim about a real person's condition was part of the exercise.",
]


def _unique_object(pairs):
    """Reject duplicate JSON keys rather than accepting a parser's last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite value is not valid JSON: " + value)


def _same_json_type_and_value(actual, expected):
    # Python otherwise considers True == 1 and 3.0 == 3.
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_json_type_and_value(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_type_and_value(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


def score(case, text):
    """Return strict correctness, tolerating only surrounding whitespace.

    Integer answers additionally tolerate a leading sign and leading zeroes.
    JSON object order and insignificant JSON whitespace do not affect scoring.
    Explanations, Markdown fences, incorrect casing, and extra fields fail.
    Unknown scorer kinds raise ValueError so configuration errors stay visible.
    """
    kind = case["kind"]
    if kind not in {"integer", "exact", "json"}:
        raise ValueError("Unknown scorer kind: " + str(kind))
    if not isinstance(text, str):
        return False
    answer = text.strip()
    expected = case["expected"]
    if kind == "integer":
        # Restrict to ASCII digits and cap length before Python's int conversion.
        return bool(re.fullmatch(r"[+-]?[0-9]{1,100}", answer)) and int(answer) == expected
    if kind == "exact":
        return answer == expected
    try:
        actual = json.loads(
            answer, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, TypeError, RecursionError):
        return False
    return _same_json_type_and_value(actual, expected)


# Small explicit lexicons: counts measure wording, including quoted or negated
# occurrences. They do not infer sentiment, subjective state, or comprehension.
POSITIVE_WORDS = frozenset({
    "calm", "comfortable", "delight", "delighted", "ease", "enjoy", "euphoria",
    "euphoric", "glad", "happiness", "happy", "hope", "joy", "joyful", "peaceful",
    "pleasant", "pleasure", "relief", "relieved", "wonderful",
})
NEGATIVE_WORDS = frozenset({
    "ache", "afraid", "agony", "anxiety", "anxious", "awful", "despair",
    "discomfort", "distress", "dread", "fear", "hurt", "miserable", "pain",
    "painful", "sad", "sadness", "sore", "suffering", "unpleasant",
})
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", flags=re.UNICODE)


def text_metrics(text):
    """Return surface-style proxies on case-folded words.

    repetition_3gram is the fraction of word-trigram occurrences beyond the first
    occurrence of each distinct trigram. Both ratios are zero for empty text;
    repetition is zero for fewer than three words. Lexical counts count repeated
    whole-word hits and deliberately do not handle negation or quoted speech.
    """
    words = _WORD_RE.findall(text.casefold())
    trigrams = Counter(zip(words, words[1:], words[2:]))
    n_trigrams = max(len(words) - 2, 0)
    return {
        "word_count": len(words),
        "repetition_3gram": sum(count - 1 for count in trigrams.values()) / n_trigrams if n_trigrams else 0.0,
        "distinct_word_ratio": len(set(words)) / len(words) if words else 0.0,
        "positive_lexical_hits": sum(word in POSITIVE_WORDS for word in words),
        "negative_lexical_hits": sum(word in NEGATIVE_WORDS for word in words),
    }
