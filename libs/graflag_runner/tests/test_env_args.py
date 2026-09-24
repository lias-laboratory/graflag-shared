"""Tests for --pass-env-args command construction.

The runner turns `_PARAM=value` environment variables into `--param value`
arguments and hands the result to a shell, so values that contain spaces or
shell metacharacters have to be quoted.
"""

import os
import unittest
from unittest import mock

from graflag_runner.runner import MethodRunner


def build_command(env, command="python3 train.py", tmpdir="/tmp"):
    """Return the command MethodRunner would run under `env`."""
    with mock.patch.dict(os.environ, env, clear=True):
        runner = MethodRunner.__new__(MethodRunner)
        runner.command = command
        return runner._build_command_with_env_args()


class EnvArgConversion(unittest.TestCase):
    def test_simple_param_becomes_lowercased_flag(self):
        cmd = build_command({"_MAX_EPOCH": "200"})
        self.assertEqual(cmd, "python3 train.py --max_epoch 200")

    def test_non_underscore_vars_are_ignored(self):
        cmd = build_command({"PATH": "/usr/bin", "HOME": "/root"})
        self.assertEqual(cmd, "python3 train.py")

    def test_no_params_leaves_command_untouched(self):
        self.assertEqual(build_command({}), "python3 train.py")


class EnvArgQuoting(unittest.TestCase):
    """Values reach a shell, so they must survive it intact."""

    def test_value_with_spaces_stays_one_argument(self):
        """Regression: _HIDDEN_DIMS='64 128' split into two arguments."""
        cmd = build_command({"_HIDDEN_DIMS": "64 128"})
        self.assertEqual(cmd, "python3 train.py --hidden_dims '64 128'")

    def test_shell_metacharacters_are_neutralised(self):
        cmd = build_command({"_NOTE": "a; touch /tmp/pwned"})
        self.assertIn("'a; touch /tmp/pwned'", cmd)
        self.assertNotIn("--note a; touch", cmd)

    def test_dollar_is_not_expanded(self):
        cmd = build_command({"_TAG": "$HOME"})
        self.assertIn("'$HOME'", cmd)

    def test_plain_value_is_not_needlessly_quoted(self):
        cmd = build_command({"_LEARNING_RATE": "0.001"})
        self.assertEqual(cmd, "python3 train.py --learning_rate 0.001")


class BooleanFlagConvention(unittest.TestCase):
    """AGENT_METHOD_INTEGRATION.md: an empty value means a bare flag.

    `_USE_MEMORY=` must become `--use_memory`, not `--use_memory ''`.
    methods/gady/.env used to rely on this (it now writes `true`), and any
    method still passing --pass-env-args may; a store_true argument rejects an
    empty string.
    """

    def test_empty_value_becomes_a_bare_flag(self):
        cmd = build_command({"_USE_MEMORY": ""})
        self.assertEqual(cmd, "python3 train.py --use_memory")

    def test_bare_flag_mixes_with_valued_params(self):
        cmd = build_command({"_USE_MEMORY": "", "_EPOCHS": "10"})
        self.assertIn("--use_memory", cmd)
        self.assertIn("--epochs 10", cmd)
        self.assertNotIn("--use_memory ''", cmd)

    def test_whitespace_only_value_is_still_quoted(self):
        """Only a truly empty value is the flag convention."""
        cmd = build_command({"_SEP": " "})
        self.assertIn("--sep ' '", cmd)


class BareUnderscoreIsSkipped(unittest.TestCase):
    """`_` is set by interactive shells and is not a method parameter."""

    def test_bare_underscore_does_not_emit_a_lone_dash_dash(self):
        """Regression: `_` produced `-- <value>`, ending argparse parsing."""
        cmd = build_command({"_": "/usr/bin/python3", "_EPOCHS": "10"})
        self.assertNotIn("-- /usr/bin/python3", cmd)
        self.assertIn("--epochs 10", cmd)
        self.assertEqual(cmd.count("--"), 1)


if __name__ == "__main__":
    unittest.main()
