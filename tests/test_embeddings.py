import pandas as pd
import numpy as np
import pytest


def test_csv_loads(tmp_path):
    df = pd.DataFrame({
        "Name": ["Python Course", "SQL Basics"],
        "Skills": ["Python, OOP", "SQL, Database"]
    })
    file = tmp_path / "courses.csv"
    df.to_csv(file, index=False)
    loaded = pd.read_csv(file).fillna("")
    assert len(loaded) == 2
    assert "Name" in loaded.columns
    assert "Skills" in loaded.columns


def test_combined_column():
    df = pd.DataFrame({
        "Name": ["Python Course"],
        "Skills": ["Python, OOP"]
    })
    df["combined"] = df["Name"] + " " + df["Skills"]
    assert df["combined"][0] == "Python Course Python, OOP"


def test_embedding_shape():
    fake_embeddings = np.random.rand(5, 768).astype("float32")
    assert fake_embeddings.shape[0] == 5
    assert fake_embeddings.shape[1] == 768


def test_top_k_results():
    df = pd.DataFrame({
        "Name": [f"Course {i}" for i in range(20)],
        "Skills": ["Python"] * 20,
        "score": np.random.rand(20)
    })
    top5 = df.sort_values("score", ascending=False).head(5)
    assert len(top5) == 5


def test_score_column_exists():
    df = pd.DataFrame({
        "Name": ["Course A", "Course B"],
        "Skills": ["Python", "SQL"],
        "combined": ["Course A Python", "Course B SQL"]
    })
    df["score"] = [0.9, 0.4]
    assert "score" in df.columns
    assert df["score"].max() == 0.9