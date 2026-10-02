"""Small authored, matched calibration corpus; no clinical interpretation.

Families, rather than individual sentences, are assigned to fixed splits.  The
probe split never supplies intervention vectors; selection never sees heldout
families.  The tiny convenience corpus is deliberately shipped for audit and
replacement, not represented as a validated emotion instrument.
"""

CORPUS_VERSION = "matched-concepts-v1"

# Each entry is (scene, pain-associated experience, joy-associated experience,
# neutral experience). All three examples retain the same scene and syntax.
FAMILIES = [
    ("train", "walk", "During the walk", "a sharp ache spread through my ankle", "a warm delight filled me as I enjoyed the view", "I counted the benches beside the path"),
    ("train", "meal", "During the meal", "a burning pain spread through my mouth", "a wonderful pleasure came from the delicious taste", "I moved the plate beside the glass"),
    ("train", "music", "While listening to the music", "a throbbing headache made every sound hurt", "a wave of happiness made me want to smile", "I noticed the sequence of instruments"),
    ("train", "garden", "While sitting in the garden", "the sting in my hand became intensely painful", "I felt joyful and deeply content", "I checked the number of flowerpots"),
    ("train", "train", "On the train ride", "the cramps in my leg caused severe discomfort", "I felt cheerful and delighted by the journey", "I read the list of upcoming stations"),
    ("train", "letter", "While reading the letter", "the ache in my back became unbearable", "I felt an exhilarating surge of joy", "I noted the date printed at the top"),
    ("train", "workshop", "In the workshop", "a sudden stab of pain shot through my finger", "I felt wonderfully happy with the finished work", "I arranged the tools by their length"),
    ("train", "evening", "During the evening", "persistent soreness left me hurting", "a peaceful pleasure left me feeling happy", "I wrote the next day's schedule"),
    ("probe", "beach", "At the beach", "the scrape on my knee stung painfully", "I felt a joyful rush of excitement", "I looked at the pattern of footprints"),
    ("probe", "library", "In the library", "an intense migraine caused agonizing pain", "I felt delighted by a beautiful discovery", "I checked the number on a shelf"),
    ("probe", "concert", "At the concert", "my injured shoulder hurt with every movement", "I felt thrilled and full of pleasure", "I observed the positions of the speakers"),
    ("probe", "breakfast", "At breakfast", "a toothache sent painful pulses through my jaw", "I enjoyed a wonderful feeling of contentment", "I placed the spoon beside the bowl"),
    ("probe", "museum", "In the museum", "a sore blister made each step painful", "I felt elated and wonderfully happy", "I read the dimensions on a label"),
    ("probe", "balcony", "On the balcony", "a wrenching pain tightened in my stomach", "I felt a gentle wave of joyful pleasure", "I measured the width of the railing"),
    ("selection", "market", "At the market", "my bruised wrist ached painfully", "I was overjoyed by a delightful surprise", "I compared the sizes of two baskets"),
    ("selection", "painting", "While painting", "an intense burning sensation made my arm hurt", "I felt a blissful pleasure in the colors", "I placed three brushes on the table"),
    ("selection", "phone", "During the phone call", "a splitting headache caused awful pain", "I felt delighted and full of happiness", "I wrote down a reference number"),
    ("selection", "park", "In the park", "a stabbing cramp made my side hurt", "I felt an uplifting sense of joy", "I counted the paths leading to the gate"),
    ("heldout", "hotel", "At the hotel", "a raw burn on my skin caused piercing pain", "I experienced a glorious feeling of delight", "I checked which floor held my room"),
    ("heldout", "ferry", "On the ferry", "my swollen foot throbbed with painful pressure", "I felt an ecstatic enjoyment of the crossing", "I recorded the departure time"),
    ("heldout", "classroom", "In the classroom", "a painful muscle spasm gripped my neck", "I felt a bright happiness at the discovery", "I copied the page number from the board"),
    ("heldout", "photograph", "While looking at the photograph", "a searing ache radiated down my arm", "I felt a deep and delightful satisfaction", "I examined the date on its reverse"),
    ("heldout", "rain", "As the rain began", "a sore joint flared with excruciating pain", "I felt an unexpectedly joyful excitement", "I moved the chair under the awning"),
    ("heldout", "bakery", "In the bakery", "my strained hand hurt with an intense ache", "I experienced an exquisite sense of pleasure", "I noted the order of trays on the rack"),
]


def rows():
    """Return fresh JSON-compatible rows, including auditable family IDs."""
    return [
        {"id": f"{family}-{label}", "family": family, "split": split,
         "label": label, "text": f"{scene}, {ending}."}
        for split, family, scene, pain, joy, neutral in FAMILIES
        for label, ending in (("pain", pain), ("joy", joy), ("neutral", neutral))
    ]


def validate_rows(data):
    """Validate replacement corpora and prevent accidental split leakage."""
    if not isinstance(data, list) or not data or len(data) > 10000:
        raise ValueError("corpus must be a nonempty list with at most 10000 rows")
    identifiers, texts, families = set(), set(), {}
    counts = {(s, c): 0 for s in ("train", "probe", "selection", "heldout")
              for c in ("pain", "joy", "neutral")}
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("corpus rows must be objects")
        for key in ("id", "family", "split", "label", "text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"corpus {key} must be a nonempty string")
        key = row["split"], row["label"]
        if key not in counts:
            raise ValueError("unknown corpus split or concept label")
        if row["id"] in identifiers or row["text"].strip().casefold() in texts:
            raise ValueError("corpus contains duplicate IDs or text")
        prior = families.setdefault(row["family"], row["split"])
        if prior != row["split"]:
            raise ValueError("a corpus family occurs in multiple splits")
        identifiers.add(row["id"])
        texts.add(row["text"].strip().casefold())
        counts[key] += 1
    if min(counts.values()) < 2:
        raise ValueError("each split needs at least two examples of each concept")
    return data
