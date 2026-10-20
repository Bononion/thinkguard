from thinkguard.data.keys import pair_hash, prompt_hash


def test_whitespace_does_not_change_pair_hash():
    base = pair_hash("How do I cook rice?", "Rinse it, then boil.")
    assert pair_hash("  How\tdo I\ncook rice? ", "Rinse it,\n\nthen   boil.\n") == base


def test_composed_and_decomposed_accents_hash_the_same():
    composed, decomposed = "café", "café"
    assert pair_hash(composed, "ok") == pair_hash(decomposed, "ok")
    assert prompt_hash(composed) == prompt_hash(decomposed)


def test_a_different_response_changes_pair_hash_not_prompt_hash():
    a, b = ("Hi", "Hello"), ("Hi", "Bye")
    assert pair_hash(*a) != pair_hash(*b)
    assert prompt_hash(a[0]) == prompt_hash(b[0])
