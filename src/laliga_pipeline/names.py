import unicodedata, re

ALIASES = {
    "ath bilbao": "athletic club",
    "athletic bilbao": "athletic club",
    "atletico madrid": "atlético madrid",
    "atletico de madrid": "atlético madrid",
    "celta": "celta de vigo",
    "vallecano": "rayo vallecano",
    "e/spanol": "rcd espanyol",
    "espanyol": "rcd espanyol",
    "mallorca": "rcd mallorca",
    "alaves": "deportivo alavés",
    "betis": "real betis",
    "sevilla": "sevilla fc",
    "valencia": "valencia cf",
    "girona": "girona fc",
    "las palmas": "ud las palmas",
    "osasuna": "ca osasuna",
    "cadiz": "cádiz cf",
    "getafe": "getafe cf",
    "villarreal": "villarreal cf",
    "valladolid": "real valladolid",
    "leganes": "cd leganés",
    "barcelona": "fc barcelona",
    "real sociedad": "real sociedad",
    "real madrid": "real madrid",
}

def normalize_team(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z0-9\s/]+", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return ALIASES.get(s, s)

#Name normalization reconciles historical naming drift and accent handling so multi‑season parameters and merges remain stable