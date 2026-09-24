"""Tests for the SUPPORTED_DATASETS compatibility check.

The runner enforces this (the orchestrator does not), so a pattern that fails
to match makes the method reject every dataset it is given.
"""

import os
import unittest
from unittest import mock

from graflag_runner.runner import MethodRunner


def accepts(dataset: str, patterns: str) -> bool:
    """True if the runner would accept `dataset` for a method declaring `patterns`."""
    env = {
        "DATA": f"/shared/datasets/{dataset}/",
        "EXP": "/tmp/exp",
        "METHOD_NAME": "m",
        "COMMAND": "true",
        "SUPPORTED_DATASETS": patterns,
    }
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch.object(MethodRunner, "__init__", lambda self, **kw: None):
        try:
            MethodRunner.from_env()
            return True
        except ValueError:
            return False


class LeadingWildcards(unittest.TestCase):
    """Regression: only a trailing '*' was understood.

    methods/example advertises `SUPPORTED_DATASETS=*_snapshot`, and addgraph,
    dynwalk and strgnn all declare it -- so all three rejected every dataset
    and could not run at all.
    """

    def test_suffix_pattern_matches(self):
        self.assertTrue(accepts("btc_alpha_snapshot", "*_snapshot"))
        self.assertTrue(accepts("uci_snapshot", "*_snapshot"))
        self.assertTrue(accepts("email_snapshot", "*_snapshot"))

    def test_suffix_pattern_still_rejects_non_matches(self):
        self.assertFalse(accepts("btc_alpha", "*_snapshot"))
        self.assertFalse(accepts("snapshot_btc", "*_snapshot"))


class PrefixWildcards(unittest.TestCase):
    def test_prefix_patterns_keep_working(self):
        self.assertTrue(accepts("bond_inj_cora", "bond_*"))
        self.assertTrue(accepts("generaldyg_btc_alpha", "generaldyg_*"))
        self.assertTrue(accepts("slade_wikipedia", "slade_*"))

    def test_prefix_pattern_rejects_non_matches(self):
        self.assertFalse(accepts("uci", "bond_*"))


class ExactAndLists(unittest.TestCase):
    def test_exact_name(self):
        self.assertTrue(accepts("uci", "uci"))
        self.assertFalse(accepts("uci2", "uci"))

    def test_comma_separated_list(self):
        patterns = "uci,btc_alpha,btc_otc,digg"
        self.assertTrue(accepts("btc_otc", patterns))
        self.assertFalse(accepts("reddit", patterns))

    def test_whitespace_around_entries_is_tolerated(self):
        self.assertTrue(accepts("digg", " uci , digg "))

    def test_mixed_exact_and_wildcard(self):
        self.assertTrue(accepts("uci", "uci,*_snapshot"))
        self.assertTrue(accepts("btc_alpha_snapshot", "uci,*_snapshot"))
        self.assertFalse(accepts("reddit", "uci,*_snapshot"))


class NoRestriction(unittest.TestCase):
    def test_empty_supported_datasets_accepts_anything(self):
        self.assertTrue(accepts("anything_at_all", ""))


class EveryShippedPatternMatchesSomething(unittest.TestCase):
    """A pattern that cannot match any dataset name is a broken method."""

    SHIPPED = {
        "bond_*": "bond_inj_cora",
        "*_snapshot": "btc_alpha_snapshot",
        "generaldyg_*": "generaldyg_btc_alpha",
        "slade_*": "slade_wikipedia",
        "gady_*": "gady_email_dnc",
        "streamspot_*": "streamspot_all",
        "anograph_*": "anograph_darpa",
        "uci,btc_alpha,btc_otc,digg": "uci",
    }

    def test_each_pattern_matches_its_dataset(self):
        for pattern, dataset in self.SHIPPED.items():
            with self.subTest(pattern=pattern):
                self.assertTrue(accepts(dataset, pattern))


if __name__ == "__main__":
    unittest.main()
