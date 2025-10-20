
import re


PROTECT_RE = re.compile(r'\[[^\]]*\]|<[^>]*>')

def _process_with_protection(text: str, processor) -> str:

    out = []
    last = 0

    def process_plain(seg: str) -> str:
        if not seg:
            return seg
        parts = []

        for m in re.finditer(r'\s+|[^\s]+', seg):
            tok = m.group(0)
            if tok.isspace():
                parts.append(tok)
            else:
                parts.append(processor(tok))
        return ''.join(parts)

    for m in PROTECT_RE.finditer(text):
        if m.start() > last:
            out.append(process_plain(text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()

    if last < len(text):
        out.append(process_plain(text[last:]))

    return ''.join(out)



EGYPTIAN_TO_LATIN_MAP = {

        'w': 'u',
        'š': 'sh',
        'z': 's',
        'ś': 's',
        'ḏ': 'dz',
        'ḥ': 'h',
        'ḫ': 'kh',
        'ẖ': 'kh',
        'ẖ': 'kh',
        'ṯ': 'tz',
        'ṱ': 'tj',
        'i̯': 'j',
        'ꞽ': 'j',
        'ꜥ': 'a',
        'ꜣ': 'qa',
        'ἰ': 'j',
        'ἱ': 'j',
        '⸗': '',
        '=': '',
}


COPTIC_TO_LATIN_MAP = {

        'ⲟⲩ': 'ou',
        'ϯ': 'tj',
        'Φ': 'ph',
        'α': 'a',
        'β': 'b',
        'γ': 'u',
        'δ': 'd',
        'ε': 'e',
        'ζ': 'z',
        'θ': 'th',
        'ι': 'j',
        'κ': 'k',
        'λ': 'l',
        'μ': 'm',
        'ν': 'n',
        'ξ': 'ks',
        'ο': 'o',
        'π': 'p',
        'ρ': 'r',
        'ς': 's',
        'σ': 's',
        'τ': 't',
        'υ': 'u',
        'φ': 'ph',
        'χ': 'kh',
        'ω': 'oo',
        'ό': 'o',
        'ϣ': 'sh',
        'ϥ': 'f',
        'ϧ': 'kh',
        'ϩ': 'h',
        'ϫ': 'dz',
        'ϭ': 'g',

        '†': 'tj',
        'ⲁ': 'a',
        'ⲃ': 'b',
        'ⲅ': 'g',
        'ⲇ': 'd',
        'Ⲉ': 'e',
        'ⲉ': 'e',
        'ⲋ': '6',
        'ⲍ': 'z',
        'ⲏ': 'ee',
        'ⲑ': 'th',
        'ⲓ': 'j',
        'ⲕ': 'k',
        'ⲗ': 'l',
        'ⲙ': 'm',
        'ⲛ': 'n',
        'ⲝ': 'ks',
        'ⲟ': 'o',
        'ⲡ': 'p',
        'ⲣ': 'r',
        'ⲥ': 's',
        'ⲧ': 't',
        'ⲩ': 'u',
        'ⲫ': 'ph',
        'ⲭ': 'kh',
        'ⲯ': 'ps',
        'ⲱ': 'oo',

}


def _augment_case_variants_for_map(d: dict):
    add = {}
    for k, v in list(d.items()):

        variants = {k.upper(), k[:1].upper() + k[1:]}
        for var in variants:
            if var != k and var not in d:
                add[var] = v
    d.update(add)

_augment_case_variants_for_map(COPTIC_TO_LATIN_MAP)

def normalize_to_latin(text: str, lang: str) -> str:

    def _proc(seg: str) -> str:
        mapping_dict = None
        if lang in ["hieroglyphic", "demotic"]:
            mapping_dict = EGYPTIAN_TO_LATIN_MAP
        elif lang in ["bohairic", "sahidic"]:
            mapping_dict = COPTIC_TO_LATIN_MAP

        if mapping_dict:

            sorted_keys = sorted(mapping_dict.keys(), key=len, reverse=True)

            normalized_text = seg
            for key in sorted_keys:
                normalized_text = normalized_text.replace(key, mapping_dict[key])
        else:

            normalized_text = seg


        normalized_text = normalized_text.replace('.', '')



        return " ".join(normalized_text.split())


    return _process_with_protection(text, _proc)




EGYPTIAN_TO_IPA_MAP = {
    'ꜣ': 'ʔ',  # Aleph -> Glottal stop
    'ꜥ': 'ʕ',  # Ayin -> Voiced pharyngeal fricative
    'y': 'j',
    'i̯': 'j',
    'ꞽ': 'j',
    'ἰ': 'j',
    'ἱ': 'j',
    'w': 'u',
    'b': 'b',
    'p': 'p',
    'f': 'f',
    'm': 'm',
    'n': 'n',
    'r': 'r',
    'h': 'h',
    'ḥ': 'ħ',
    'ḫ': 'χ',
    'ẖ': 'ç',
    's': 's',
    'z': 's',
    'š': 'ʃ',
    'q': 'q',
    'k': 'k',
    'g': 'g',
    't': 't',
    'ṯ': 'tʃ',
    'd': 'd',
    'ḏ': 'ɟ',
    '⸗': '',
    '=': '',
}


COPTIC_TO_IPA_RULES = [

    (re.compile('ⲟⲩ'), 'u'),
    (re.compile('ⲉⲓ'), 'i'),
    (re.compile('ⲝ'), 'ks'),
    (re.compile('ⲯ'), 'ps'),
    (re.compile('ϯ'), 'ti'),
    (re.compile('ⲭ'), 'kʰ'),
    (re.compile('ⲑ'), 'tʰ'),
    (re.compile('ⲫ'), 'pʰ'),
    (re.compile('ϣ'), 'ʃ'),
    (re.compile('ϥ'), 'f'),
    (re.compile('ϧ'), 'x'),
    (re.compile('ϩ'), 'h'),
    (re.compile('ϫ'), 'ɟ'),
    (re.compile('ϭ'), 'tʃ'),

    (re.compile('ⲁ'), 'a'),
    (re.compile('ⲃ'), 'b'),
    (re.compile('ⲅ'), 'g'),
    (re.compile('ⲇ'), 'd'),
    (re.compile('ⲉ'), 'ə'),
    (re.compile('ⲍ'), 'z'),
    (re.compile('ⲏ'), 'eː'),
    (re.compile('ⲓ'), 'i'),
    (re.compile('ⲕ'), 'k'),
    (re.compile('ⲗ'), 'l'),
    (re.compile('ⲙ'), 'm'),
    (re.compile('ⲛ'), 'n'),
    (re.compile('ⲟ'), 'o'),
    (re.compile('ⲡ'), 'p'),
    (re.compile('ⲣ'), 'r'),
    (re.compile('ⲥ'), 's'),
    (re.compile('ⲧ'), 't'),
    (re.compile('ⲩ'), 'u'),
    (re.compile('ⲱ'), 'oː'),
]


def _make_rules_ignore_case(rules):
    new = []
    for pat, rep in rules:

        if pat.flags & re.IGNORECASE:
            new.append((pat, rep))
        else:
            new.append((re.compile(pat.pattern, pat.flags | re.IGNORECASE), rep))
    return new

COPTIC_TO_IPA_RULES = _make_rules_ignore_case(COPTIC_TO_IPA_RULES)


def _egyptian_insert_schwa(ipa_text: str) -> str:
    words = ipa_text.split()
    if not words:
        return ipa_text

    LONG_TOKENS = ['tʃ', 'dʒ']

    symbols_in_text = {ch for ch in ipa_text if ch.strip()}
    for v in EGYPTIAN_TO_IPA_MAP.values():
        symbols_in_text.update(list(v))


    PUNCTS = {'.'}

    symbols_in_text -= PUNCTS
    SINGLE_TOKENS = sorted({c for c in symbols_in_text if len(c) == 1})

    tok_pattern = re.compile(
        r'(?:' + '|'.join(map(re.escape, LONG_TOKENS)) +
        r'|[' + ''.join(map(re.escape, SINGLE_TOKENS)) + r'])'
    )

    BLOCKERS = {'j', 'ʔ', 'ʕ', 'w', 'a', 'e', 'i', 'o', 'u', 'ə'}

    out_words = []
    for w in words:

        w_clean = w.replace('.', '')

        # ★ 新：按连字符切分并保留分隔符本身
        parts = re.split(r'(-)', w_clean)

        seg_outs = []
        for part in parts:
            if part == '-':
                seg_outs.append(part)
                continue

            seg = part
            tokens = tok_pattern.findall(seg)
            if not tokens:
                seg_outs.append(seg)
                continue

            if len(tokens) == 1:

                if tokens[0] in BLOCKERS:
                    seg_outs.append(tokens[0])
                else:
                    seg_outs.append('e' + tokens[0])
                continue

            new_seq = [tokens[0]]
            for i in range(len(tokens) - 1):
                left, right = tokens[i], tokens[i + 1]
                if left in BLOCKERS or right in BLOCKERS:
                    pass
                else:
                    new_seq.append('e')
                new_seq.append(right)
            seg_outs.append(''.join(new_seq))

        out_words.append(''.join(seg_outs))

    return ' '.join(out_words)


import re, unicodedata

def _sub_by_map(text: str, mapping: dict, norm: str = "NFC") -> str:

    if norm:
        text = unicodedata.normalize(norm, text)
        mapping = {unicodedata.normalize(norm, k): v for k, v in mapping.items()}

    if not mapping:
        return text

    pattern = re.compile("|".join(sorted(map(re.escape, mapping.keys()), key=len, reverse=True)))
    return pattern.sub(lambda m: mapping[m.group(0)], text)

def normalize_to_ipa(text: str, lang: str) -> str:

    def _proc(seg: str) -> str:
        if lang in ["hieroglyphic", "demotic"]:
            normalized_text = _sub_by_map(seg, EGYPTIAN_TO_IPA_MAP, norm="NFC")
            normalized_text = _egyptian_insert_schwa(normalized_text)
        elif lang in ["bohairic", "sahidic"]:
            normalized_text = seg
            for pattern, replacement in COPTIC_TO_IPA_RULES:
                normalized_text = pattern.sub(replacement, normalized_text)
        else:
            normalized_text = seg

        normalized_text = normalized_text.replace('.', '')

        return " ".join(normalized_text.split())

    return _process_with_protection(text, _proc)




def normalize_text(text: str, lang: str, mode: str) -> str:
    if mode.upper() == "LATIN":
        return normalize_to_latin(text, lang)
    elif mode.upper() == "IPA":
        return normalize_to_ipa(text, lang)
    elif mode.upper() == "NONE" or mode is None:
        return text
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


if __name__ == "__main__":


    test_cases = [

        ("sš nfr.t m ẖr.w ⸗f", "hieroglyphic"),
        ("pꜣ-dy-wsꞽr-m-jwn", "demotic"),
        ("šꜥ-ḏ.t", "demotic"),


        ("ⲛⲟⲩϥⲉ", "bohairic"),
        ("ϣⲁⲓ", "sahidic"),
        ("ϩⲟⲣⲓ", "bohairic"),
        ("ϫⲓϫ", "sahidic"),


        ("This is a beautiful test.", "english")
    ]

    print("\n" + "=" * 80)
    print("🚀 STARTING NORMALIZATION TEST 🚀")
    print("=" * 80)

    for original_text, lang in test_cases:
        print(f"\n--- Testing Case ---")
        print(f"Original Text: '{original_text}' ({lang})")


        latin_normalized = normalize_text(original_text, lang, "LATIN")
        print(f"  -> LATIN Mode: '{latin_normalized}'")


        ipa_normalized = normalize_text(original_text, lang, "IPA")
        print(f"  -> IPA Mode  : '{ipa_normalized}'")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE ✅")
    print("=" * 80)

    print("\n=== IPA Demo (fill your xxx strings) ===")

    samples = {
        "hieroglyphic": "<hiero> [gap] pꜣy-dy.t-ꜣs.t [gap] [gap] [gap]",
        "demotic": "n-ḏr.t nꜣ-wꜣḥ-ꞽb-rꜥ ḥwy ⸗y r-bnr-r nꜣ ꜥ.wy.w nꜣy ⸗y ꞽt.ṱ.w r-ḏbꜣ msty ⸗y",
        "sahidic": "Ⲫⲛ ϣⲁϫⲉ ⲙ ⲡⲁⲓ ⲉⲧ ϫⲱ ⲛ ⲛⲁⲓ ϩⲉⲛ ϭⲟⲗ ⲛⲉ",
        "bohairic": "ⲧⲟⲧⲉ ⲉϥⲉ ⲟⲩⲱⲧⲉⲃ ⲛϫⲉ ⲡⲓ ⲡⲛⲉⲩⲙⲁ ⲟⲩⲟϩ ⲉϥⲉ ⲥⲓⲛⲓ ⲟⲩⲟϩ ⲉϥⲉ ⲭⲱ ⲉⲃⲟⲗ ⲑⲁⲓ ⲡⲉ ϯ ϫⲟⲙ ⲛⲧⲉ ⲡⲁ ⲛⲟⲩϯ",

    }

    for lg, raw in samples.items():
        ipa = normalize_text(raw, lg, "IPA")
        print(f"[{lg}]  raw: {raw}  ->  IPA: {ipa}")

    for raw in ["nfr", "mnys", "mnyst", "my","mny", "ꞽn", "pꜣ-dy-wsꞽr"]:
        print(f"[hieroglyphic] {raw} -> {normalize_text(raw, 'hieroglyphic', 'IPA')}")

    print("\n=== LATIN Demo (fill your xxx strings) ===")

    samples = {
        "hieroglyphic": "<hiero> [gap] pꜣy-dy.t-ꜣs.t [gap] [gap] [gap]",
        "demotic": "n-ḏr.t nꜣ-wꜣḥ-ꞽb-rꜥ ḥwy ⸗y r-bnr-r nꜣ ꜥ.wy.w nꜣy ⸗y ꞽt.ṱ.w r-ḏbꜣ msty ⸗y",
        "sahidic": "Ⲫⲛ ϣⲁϫⲉ ⲙ ⲡⲁⲓ ⲉⲧ ϫⲱ ⲛ ⲛⲁⲓ ϩⲉⲛ ϭⲟⲗ ⲛⲉ",
        "bohairic": "ⲧⲟⲧⲉ ⲉϥⲉ ⲟⲩⲱⲧⲉⲃ ⲛϫⲉ ⲡⲓ ⲡⲛⲉⲩⲙⲁ ⲟⲩⲟϩ ⲉϥⲉ ⲥⲓⲛⲓ ⲟⲩⲟϩ ⲉϥⲉ ⲭⲱ ⲉⲃⲟⲗ ⲑⲁⲓ ⲡⲉ ϯ ϫⲟⲙ ⲛⲧⲉ ⲡⲁ ⲛⲟⲩϯ",

    }

    for lg, raw in samples.items():
        latin = normalize_text(raw, lg, "LATIN")
        print(f"[{lg}]  raw: {raw}  ->  LATIN: {latin}")


    for raw in ["nfr", "mnys", "mnyst", "my", "mny", "ꞽn", "pꜣ-dy-wsꞽr"]:
        print(f"[hieroglyphic] {raw} -> {normalize_text(raw, 'hieroglyphic', 'LATIN')}")
