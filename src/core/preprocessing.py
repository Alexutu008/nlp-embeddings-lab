import re

def simple_tokeninze(text: str) -> list[str]:
    text = text.lower().strip()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens