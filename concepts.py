"""Query understanding: turn everyday wording into the terms regulations actually use.

IMPORTANT: this map is HAND-WRITTEN. Every entry must be checked against the real document
before you rely on it. "definition" is a general explanation, NOT text from the regulation,
and the app labels it that way.

What this module does:
  1. normalize()      - fixes likely spelling mistakes by snapping words to known vocabulary
  2. detect_concepts  - finds which regulatory concepts the question is about
  3. analyze_query    - returns concepts, extra search terms, and details that are missing
"""
import difflib
import re

CONCEPTS = [
    {
        "name": "Setback / margin (open space around the building)",
        "triggers": ["setback", "set back", "margin", "empty space", "open space", "space between",
                     "leave space", "how close can i", "how far must", "how far should", "how far from my",
                     "distance from my", "distance from the", "edge of my land",
                     "edge of my plot", "edge of the plot", "edge of land", "right up to",
                     "boundary", "neighbour", "neighbor", "side space", "back yard", "front yard",
                     "clearance"],
        "terms": ["margin", "setback", "set back", "side margin", "rear margin",
                  "road side margin", "open to sky"],
        "definition": "The open space that must be left between a building and the boundaries of its plot (front, sides and rear).",
        "factors": ["plot size", "road width", "building height", "building use"],
    },
    {
        "name": "Ground coverage",
        "triggers": ["ground coverage", "plot coverage", "how much of my land", "how much of the plot",
                     "cover with", "covered by", "land covered", "footprint", "built on the plot",
                     "how much can i build on"],
        "terms": ["ground coverage", "plot coverage", "built up area", "coverage"],
        "definition": "The share of the plot area that the building's footprint is allowed to cover.",
        "factors": ["plot size", "building use"],
    },
    {
        "name": "Floor space index (FSI)",
        "triggers": ["fsi", "floor space index", "floor area ratio", "total built up", "how much area can i build",
                     "construction area allowed", "how much can i build in total", "built up area"],
        "terms": ["floor space index", "FSI", "chargeable FSI", "built up area"],
        "definition": "The ratio of the total floor area a building may have to the area of its plot.",
        "factors": ["plot size", "road width", "building use"],
    },
    {
        "name": "Building height",
        "triggers": ["another floor", "extra floor", "add a floor", "add floor", "more floors", "how many floors",
                     "how tall", "how high", "height", "storey", "storeys", "story", "stories", "upper floor"],
        "terms": ["building height", "permissible height", "height limit", "number of storeys"],
        "definition": "The maximum height or number of storeys a building may have.",
        "factors": ["road width", "plot size", "building use"],
    },
    {
        "name": "Parking",
        "triggers": ["parking", "park my car", "car space", "car spot", "vehicles", "garage", "parking space"],
        "terms": ["parking", "parking space", "parking requirement", "basement parking"],
        "definition": "The number and size of vehicle parking spaces a building must provide.",
        "factors": ["building use", "plot size"],
    },
    {
        "name": "Permitted land use (zoning)",
        "triggers": ["open a shop", "run a shop", "start a business", "shop at home", "office at home",
                     "factory", "commercial", "use my land", "use the land", "permitted use", "allowed to run",
                     "business from home", "zone", "zoning"],
        "terms": ["permissible use", "land use", "use zone", "residential zone", "commercial use", "permitted uses"],
        "definition": "Which activities (housing, shops, offices, industry and so on) are allowed on land in a given zone.",
        "factors": ["building use", "city or authority"],
    },
    {
        "name": "Basement",
        "triggers": ["basement", "cellar", "underground floor", "underground parking"],
        "terms": ["basement", "podium", "basement parking"],
        "definition": "A floor built partly or wholly below ground level.",
        "factors": ["plot size", "building use"],
    },
    {
        "name": "Road width and building line",
        "triggers": ["road widening", "building line", "road width", "how close to the road", "near the road",
                     "next to the road", "from the road", "to the road", "close to the road",
                     "road boundary", "road side"],
        "terms": ["road width", "building line", "road boundary", "road widening", "road side margin"],
        "definition": "Rules that depend on the width of the road in front of a plot, and the line beyond which building is not allowed.",
        "factors": ["road width", "plot size"],
    },
    {
        "name": "Fire safety, stairs and lifts",
        "triggers": ["fire exit", "fire escape", "fire safety", "lift", "elevator", "staircase", "stairs",
                     "stairway", "escape route", "emergency exit"],
        "terms": ["fire escape", "fire safety", "staircase", "lift", "means of escape"],
        "definition": "Requirements for safe exit, stairs and lifts in buildings.",
        "factors": ["building height", "building use"],
    },
    {
        "name": "Rainwater harvesting",
        "triggers": ["rain water", "rainwater", "water harvesting", "groundwater", "collect rain"],
        "terms": ["rainwater harvesting", "recharge", "groundwater recharge"],
        "definition": "Collecting and storing rainwater, or letting it soak into the ground.",
        "factors": ["plot size"],
    },
    {
        "name": "Permission and approval",
        "triggers": ["permission", "approval", "approve", "permit", "sanction", "licence", "license",
                     "building plan approval", "how long to approve"],
        "terms": ["development permission", "building permission", "application", "sanction"],
        "definition": "The process and conditions for getting official permission to build.",
        "factors": ["city or authority"],
    },
]

FACTOR_PATTERNS = {
    "plot size": r"\d+\s*(sq|square|m2|sqm|sq\.?\s?m|yard|gaj)|plot size|plot area|square",
    "road width": r"road.*\d+|\d+\s*(m|metre|meter|ft|feet)\b.*road|road width|wide road|narrow road",
    "building height": r"\d+\s*(storey|story|floor|m\b|metre|meter)|height|storeys|floors",
    "building use": r"residential|commercial|industrial|shop|office|school|house|flat|apartment|factory|hospital|home",
    "city or authority": r"ahmedabad|surat|rajkot|vadodara|gandhinagar|auda|suda|city|municipal|authority",
}

_WORD = re.compile(r"[a-z]+")
_VOCAB = sorted({w for c in CONCEPTS for t in c["triggers"] for w in _WORD.findall(t) if len(w) >= 5})


def normalize(query):
    """Lowercase, keep letters only, and snap near-miss spellings to known vocabulary."""
    out = []
    for w in _WORD.findall(query.lower()):
        if len(w) >= 5 and w not in _VOCAB:
            match = difflib.get_close_matches(w, _VOCAB, n=1, cutoff=0.84)
            if match:
                w = match[0]
        out.append(w)
    return " ".join(out)


def detect_concepts(query):
    norm = " " + normalize(query) + " "
    found = []
    for c in CONCEPTS:
        if any(" " + t + " " in norm for t in c["triggers"]):
            found.append(c)
    return found


def analyze_query(query):
    found = detect_concepts(query)
    terms, factors = [], []
    for c in found:
        terms += [t for t in c["terms"] if t not in terms]
        factors += [f for f in c["factors"] if f not in factors]
    low = query.lower()
    missing = [f for f in factors if not re.search(FACTOR_PATTERNS[f], low)]
    return {
        "concepts": [c["name"] for c in found],
        "definitions": [(c["name"], c["definition"]) for c in found],
        "terms": terms,
        "expanded": (query + " " + " ".join(terms)).strip(),
        "missing_details": missing,
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Can I build my house right up to the edge of my land?"
    print(analyze_query(q))
