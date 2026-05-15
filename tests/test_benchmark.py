"""
Tests for the benchmark runner.

Validates report structure, JSON output, and Markdown file generation
using a mocked pipeline to avoid real network/model calls.
"""

from __future__ import annotations

import json
import os

import pytest

from benchmark import run_benchmark, save_json, save_markdown
from src.retrieval import RetrievalResult
from src.storage import Document


def _make_mock_pipeline(populated_pipeline):
    return populated_pipeline


class TestRunBenchmark:

    def test_returns_list(self, populated_pipeline):
        queries = ["How does the system handle peak load?"]
        report = run_benchmark(queries, top_k=3, pipeline=populated_pipeline)
        assert isinstance(report, list)

    def test_report_length_matches_queries(self, populated_pipeline):
        queries = ["query one", "query two", "query three"]
        report = run_benchmark(queries, top_k=3, pipeline=populated_pipeline)
        assert len(report) == 3

    def test_report_entry_has_required_keys(self, populated_pipeline):
        report = run_benchmark(["test query"], top_k=3, pipeline=populated_pipeline)
        entry = report[0]
        for key in ("query", "expanded_query", "rerank_occurred", "strategy_a", "strategy_b"):
            assert key in entry, f"Missing key: {key}"

    def test_strategy_a_top_k(self, populated_pipeline, small_corpus):
        report = run_benchmark(["test"], top_k=3, pipeline=populated_pipeline)
        assert len(report[0]["strategy_a"]) == min(3, len(small_corpus))

    def test_strategy_b_top_k(self, populated_pipeline, small_corpus):
        report = run_benchmark(["test"], top_k=3, pipeline=populated_pipeline)
        assert len(report[0]["strategy_b"]) == min(3, len(small_corpus))

    def test_expanded_query_present(self, populated_pipeline):
        report = run_benchmark(["peak load"], top_k=1, pipeline=populated_pipeline)
        assert isinstance(report[0]["expanded_query"], str)
        assert len(report[0]["expanded_query"]) > 0

    def test_rerank_occurred_is_bool(self, populated_pipeline):
        report = run_benchmark(["test"], top_k=3, pipeline=populated_pipeline)
        assert isinstance(report[0]["rerank_occurred"], bool)

    def test_result_dicts_have_score(self, populated_pipeline):
        report = run_benchmark(["test"], top_k=1, pipeline=populated_pipeline)
        assert "score" in report[0]["strategy_a"][0]
        assert "score" in report[0]["strategy_b"][0]


class TestSaveJson:

    def test_creates_file(self, tmp_path, populated_pipeline):
        report = run_benchmark(["test"], top_k=1, pipeline=populated_pipeline)
        out = str(tmp_path / "out.json")
        save_json(report, out)
        assert os.path.exists(out)

    def test_valid_json(self, tmp_path, populated_pipeline):
        report = run_benchmark(["test"], top_k=1, pipeline=populated_pipeline)
        out = str(tmp_path / "out.json")
        save_json(report, out)
        with open(out) as f:
            data = json.load(f)
        assert "results" in data
        assert "generated_at" in data
        assert "corpus_size" in data

    def test_results_count(self, tmp_path, populated_pipeline):
        queries = ["q1", "q2"]
        report = run_benchmark(queries, top_k=1, pipeline=populated_pipeline)
        out = str(tmp_path / "out.json")
        save_json(report, out)
        with open(out) as f:
            data = json.load(f)
        assert len(data["results"]) == 2


class TestSaveMarkdown:

    def test_creates_file(self, tmp_path, populated_pipeline):
        report = run_benchmark(["test"], top_k=1, pipeline=populated_pipeline)
        out = str(tmp_path / "benchmark.md")
        save_markdown(report, out, top_k=1)
        assert os.path.exists(out)

    def test_contains_strategy_headers(self, tmp_path, populated_pipeline):
        report = run_benchmark(["test"], top_k=1, pipeline=populated_pipeline)
        out = str(tmp_path / "benchmark.md")
        save_markdown(report, out, top_k=1)
        with open(out) as f:
            content = f.read()
        assert "Strategy A" in content
        assert "Strategy B" in content

    def test_contains_query(self, tmp_path, populated_pipeline):
        query = "How does the system handle peak load?"
        report = run_benchmark([query], top_k=1, pipeline=populated_pipeline)
        out = str(tmp_path / "benchmark.md")
        save_markdown(report, out, top_k=1)
        with open(out) as f:
            content = f.read()
        assert query in content

    def test_contains_analysis_section(self, tmp_path, populated_pipeline):
        report = run_benchmark(["test"], top_k=1, pipeline=populated_pipeline)
        out = str(tmp_path / "benchmark.md")
        save_markdown(report, out, top_k=1)
        with open(out) as f:
            content = f.read()
        assert "Analysis" in content
