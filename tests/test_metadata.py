from __future__ import annotations

from mdtodb import (
    DEFAULT_STOPWORDS,
    filetype_for,
    keywords_for,
    metadata_for,
    normalize_keyword,
    person_for,
)


def test_keywords_full_example():
    rel = "personnes/Estelle/papiers/Allemagne/ReleveIntegral_ROY_ESTELLE_2014_26-01-2021.pdf"
    assert keywords_for(rel) == [
        "personnes",
        "estelle",
        "papiers",
        "allemagne",
        "releveintegral",
        "roy",
        "2014",
        "2021",
    ]


def test_keywords_multi_segment():
    assert keywords_for("/word1 word2/word3 - word4/word5.txt") == [
        "word1",
        "word2",
        "word3",
        "word4",
        "word5",
    ]


def test_keywords_accents_and_year():
    assert keywords_for("Téléphonie - Free /Free Estelle 2024.pdf") == [
        "telephonie",
        "free",
        "estelle",
        "2024",
    ]


def test_date_year_extraction():
    assert "2021" in keywords_for("doc 26-01-2021.pdf")
    assert "2021" in keywords_for("doc 2021-01-26.pdf")
    assert "2024" in keywords_for("facture 01/2024.pdf")
    assert "2024" in keywords_for("releve 20240315.pdf")


def test_dates_adjacent_to_underscores_or_letters():
    assert "2021" in keywords_for("scan_20210503.pdf")
    assert "2021" in keywords_for("x_2021-05-03_y.pdf")
    assert "2021" in keywords_for("ESTELLE_26-01-2021.pdf")
    assert keywords_for("doc 123456789.pdf") == ["doc"]  # 9 digits, no year


def test_numbers_dropped_years_kept():
    assert keywords_for("doc 123 42 2024.pdf") == ["doc", "2024"]
    assert keywords_for("doc 1899 2100.pdf") == ["doc"]


def test_stopwords_removed():
    assert keywords_for("Le contrat de la maison.pdf") == ["contrat", "maison"]
    assert "personnes" not in DEFAULT_STOPWORDS


def test_extra_stopwords():
    assert keywords_for("Le contrat maison.pdf", stopwords=["maison"]) == ["contrat"]
    # replace mode: defaults no longer apply
    assert keywords_for("Le contrat maison.pdf", stopwords=["contrat"], replace_stopwords=True) == [
        "le",
        "maison",
    ]


def test_normalize_keyword():
    assert normalize_keyword("Téléphonie ÉTÉ") == "telephonie ete"


def test_person():
    assert person_for("personnes/Estelle/papiers/x.pdf") == "Estelle"
    assert person_for("a/Personnes/b/c/x.pdf") == "b"
    assert person_for("docs/x.pdf") is None
    assert person_for("personnes/x.pdf") is None  # personnes introduces the file itself
    assert person_for("a/personnes") is None


def test_filetype():
    assert filetype_for("a/b.PDF") == "pdf"
    assert filetype_for("a/b") == ""


def test_metadata_for_omits_empty():
    meta = metadata_for("123/45.pdf")
    assert meta["file"] == "123/45.pdf"
    assert meta["markdown"] == "123/45.md"
    assert meta["filetype"] == "pdf"
    assert "person" not in meta
    assert "keywords" not in meta  # Chroma refuses empty arrays
    assert "engine" not in meta
    assert "indexed_at" not in meta


def test_metadata_for_full():
    meta = metadata_for(
        "personnes/E/p/x.pdf",
        engine="marker",
        fingerprint="abc",
        md_sha256="def",
        embedding="default",
        indexed_at=1.5,
    )
    assert meta["person"] == "E"
    assert meta["engine"] == "marker"
    assert meta["fingerprint"] == "abc"
    assert meta["md_sha256"] == "def"
    assert meta["embedding"] == "default"
    assert meta["indexed_at"] == 1.5
    assert meta["keywords"] == ["personnes", "e", "p", "x"] or "personnes" in meta["keywords"]
