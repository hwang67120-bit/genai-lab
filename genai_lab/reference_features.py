"""Bounded tag routing, not a semantic model. Unknown tags require manual review."""
from genai_lab.reference_tag_policy import normalize_tag, gender_tag_kind, excluded_garment_tag

CHARACTER_NOUNS = frozenset((
    'hair', 'eyes', 'ears', 'tail', 'tails', 'wings', 'horns', 'fur', 'skin',
    'fangs', 'pupils', 'eyebrows', 'eyelashes',
))
CHARACTER_TERMS = frozenset((
    'bangs', 'ponytail', 'twintails', 'braid', 'braids', 'bob cut', 'ahoge',
    'sidelocks', 'sideburns', 'short hair with long locks',
    'hair between eyes', 'hair over one eye', 'hair over eyes',
    'freckles', 'scar', 'facial mark', 'facial markings', 'pointy ears',
    'heterochromia', 'androgynous', 'slender', 'muscular', 'beard',
))
GARMENT_NOUNS = frozenset((
    'shirt', 'jacket', 'blazer', 'coat', 'dress', 'gown', 'skirt', 'pants',
    'trousers', 'jeans', 'shorts', 'leggings', 'top', 'suit',
    'vest', 'sweater', 'cardigan', 'hoodie', 'uniform', 'blouse', 'cape',
    'cloak', 'bodysuit', 'leotard', 'swimsuit', 'bikini', 'bra', 'underwear',
    'panties', 'briefs', 'apron', 'kimono', 'yukata',
    'armor', 'footwear', 'shoes', 'boots', 'heels', 'loafers', 'sneakers',
    'socks', 'stockings',
    'thighhighs', 'pantyhose', 'tights', 'gloves', 'sleeves', 'collar',
    'necktie', 'bowtie', 'belt', 'buttons', 'zipper', 'pockets', 'frills',
    'lace', 'ribbon', 'bow', 'hat', 'cap', 'scarf', 'suspenders', 'cuffs',
))
GARMENT_TERMS = frozenset((
    'formal', 'pinstripes', 'plaid', 'polka dot', 'striped', 'sleeveless',
    'denim', 'leather', 'knit', 'satin', 'silk', 'embroidery', 'pleated skirt',
))
CONTEXT_TERMS = frozenset((
    'solo', 'solo focus', 'virtual youtuber', 'watermark', 'signature', 'text',
    'group', 'couple', 'cropped', 'out of frame', 'head out of frame',
    'girl', 'boy', 'girls', 'boys', 'multiple others', '1other',
))
SCENE_TERMS = frozenset((
    'full body', 'upper body', 'portrait', 'standing', 'sitting', 'lying',
    'front view', 'back view', 'simple background', 'head to toe', 'feet visible',
))


def tag_scope(tag):
    tag = normalize_tag(tag)
    if not tag:
        return 'unknown'
    if gender_tag_kind(tag) is not None or tag in CONTEXT_TERMS:
        return 'context'
    if tag in SCENE_TERMS or tag.endswith(' background'):
        return 'scene'
    words = tag.split()
    # Hair accessories are not native hair/eye/body features.
    if words[-1] in GARMENT_NOUNS or tag in GARMENT_TERMS:
        return 'garment'
    if words[-1] in CHARACTER_NOUNS or tag in CHARACTER_TERMS:
        return 'character'
    return 'unknown'


def automatic_feature_selection(candidates, scope):
    if scope not in ('character', 'garment'):
        raise ValueError('특징 출처는 캐릭터 또는 의상이어야 합니다.')
    selected, excluded, unresolved = [], [], []
    by_name = {normalize_tag(tag.tag_name): tag for tag in candidates}
    for candidate in candidates:
        if scope == 'garment' and excluded_garment_tag(candidate.tag_name):
            excluded.append(candidate.tag_name)
            continue
        actual = tag_scope(candidate.tag_name)
        if actual == scope:
            selected.append(candidate.tag_name)
        elif actual == 'unknown':
            unresolved.append(candidate.tag_name)
        else:
            excluded.append(candidate.tag_name)
    # Do not turn close alternatives into a confident species/length decision.
    if scope == 'character':
        groups = [
            {'short hair', 'long hair', 'very long hair'},
            {f'{animal} ears' for animal in ('cat', 'dog', 'fox', 'wolf', 'raccoon', 'bear', 'rabbit')},
            {f'{animal} tail' for animal in ('cat', 'dog', 'fox', 'wolf', 'raccoon', 'bear', 'rabbit')},
        ]
        for group in groups:
            found = sorted((by_name[name] for name in group if name in by_name),
                           key=lambda tag: -tag.score)
            if len(found) > 1:
                ambiguous = found if found[0].score - found[1].score < .10 else found[1:]
                for tag in ambiguous:
                    if tag.tag_name in selected:
                        selected.remove(tag.tag_name)
                        unresolved.append(tag.tag_name)
    return {'selected': tuple(dict.fromkeys(selected)),
            'excluded': tuple(dict.fromkeys(excluded)),
            'unresolved': tuple(dict.fromkeys(unresolved))}

