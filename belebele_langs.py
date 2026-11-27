"""Languages for the Belebele extension, with each family's published training-data amounts.

Codes are FLORES-200 codes, which Belebele uses too. belebele.py evaluates only the codes the
Belebele dataset actually has, so a code listed here but missing from Belebele is skipped.

XGLM: tokens in millions from Table A10 of Lin et al. (2022), the counts Paper 1 used, and the
sampling ratio with low-resource upsampling from the XGLM-7.5B model card on Hugging Face
("Training Data Statistics", commit 66264f4). The ratio is what training actually drew from each
language, including the upsampling that Lin et al. report was mis-set for ko, fi, th, bg, ca, hi
and et.

BLOOM: bytes per language from Table 1 of BigScience Workshop (2022). Chinese is Simplified plus
Traditional, as in Paper 1. Three printed typos are read as numbers: Akan "70,1554" = 701,554;
Igbo "14078,521" = 14,078,521; Telugu "2993407,159" = 2,993,407,159.
"""

XGLM = {  # code: (tokens in millions, sampled ratio)
    "eng_Latn": (803527, 0.3259), "rus_Cyrl": (147792, 0.0602), "zho_Hans": (132770, 0.0483),
    "deu_Latn": (89224, 0.0363), "spa_Latn": (87303, 0.0353), "fra_Latn": (77420, 0.0313),
    "jpn_Jpan": (66054, 0.0269), "ita_Latn": (41930, 0.0171), "por_Latn": (36586, 0.0297),
    "ell_Grek": (28762, 0.0233), "kor_Hang": (20002, 0.0811), "fin_Latn": (16804, 0.0681),
    "ind_Latn": (15424, 0.0125), "tur_Latn": (12413, 0.0101), "arb_Arab": (12249, 0.0099),
    "vie_Latn": (11199, 0.0091), "tha_Thai": (10842, 0.0440), "bul_Cyrl": (9704, 0.0393),
    "cat_Latn": (7076, 0.0287), "hin_Deva": (3448, 0.0140), "est_Latn": (3287, 0.0133),
    "ben_Beng": (1627, 0.0066), "tam_Taml": (1477, 0.0060), "urd_Arab": (1352, 0.0055),
    "swh_Latn": (908, 0.0037), "tel_Telu": (689, 0.0028), "eus_Latn": (105, 0.0043),
    "mya_Mymr": (101, 0.0030), "hat_Latn": (87, 0.0035), "quy_Latn": (3, 0.0001),
}

BLOOM = {  # code: bytes
    "eng_Latn": 484953009124, "fra_Latn": 208242620434, "spa_Latn": 175098365045,
    "por_Latn": 79277543375, "arb_Arab": 74854900600, "zho_Hans": 261781923042,
    "vie_Latn": 43709279959, "hin_Deva": 24622119985, "ind_Latn": 19972325222,
    "ben_Beng": 18606823104, "cat_Latn": 17792493289, "tam_Taml": 7989206220,
    "mal_Mlym": 3662571498, "tel_Telu": 2993407159, "urd_Arab": 2781329959,
    "npi_Deva": 2551307393, "eus_Latn": 2360470848, "kan_Knda": 2098453560,
    "mar_Deva": 1775483122, "pan_Guru": 1572109752, "guj_Gujr": 1199986460,
    "ory_Orya": 1157100133, "asm_Beng": 291522098, "swh_Latn": 236482543,
    "yor_Latn": 89695835, "kin_Latn": 40428299, "xho_Latn": 14304074, "ibo_Latn": 14078521,
    "zul_Latn": 8511561, "sna_Latn": 6638639, "lug_Latn": 4568367, "wol_Latn": 3606973,
    "run_Latn": 3272550, "fon_Latn": 2478546, "nso_Latn": 1764506, "lin_Latn": 1650804,
    "tsn_Latn": 1502200, "twi_Latn": 1265041, "nya_Latn": 1187405, "sot_Latn": 751034,
    "tso_Latn": 707634, "aka_Latn": 701554, "bam_Latn": 391747, "kik_Latn": 359615,
    "tum_Latn": 170360,
}

# Scripts written without spaces between words: fertility and STRR are undefined, as in Paper 1.
NO_WORDS_SCRIPTS = {"Hans", "Hant", "Jpan", "Thai", "Mymr", "Khmr", "Laoo", "Tibt"}

EVAL = sorted(set(XGLM) | set(BLOOM))      # every model is run on all of these
PILOT = ["eng_Latn", "fra_Latn", "hin_Deva", "ind_Latn", "swh_Latn", "tel_Telu"]


def has_words(code):
    return code.split("_")[1] not in NO_WORDS_SCRIPTS


def family(tag):
    return "bloom" if tag.startswith("bloom") else "xglm"


def seen(tag, code):
    return code in (BLOOM if family(tag) == "bloom" else XGLM)
