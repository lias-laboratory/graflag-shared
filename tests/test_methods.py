"""Validation for every method definition under `methods/`.

These are the checks that would otherwise only surface when a build or a run
fails on the cluster, minutes later: a malformed `.env`, a Dockerfile whose
COPY paths do not match the build context, a `SUPPORTED_DATASETS` pattern that
cannot match anything, an integration script that never writes a result.

Run from the repository root::

    PYTHONPATH=libs python3 -m unittest discover -s tests -v
"""

import ast
import dataclasses
import fnmatch
import pathlib
from pathlib import Path
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
METHODS = ROOT / "methods"
DATASETS = ROOT / "datasets"
IMAGES = ROOT / "images"

REQUIRED_KEYS = {"METHOD_NAME", "DESCRIPTION", "SOURCE_CODE", "COMMAND"}
KNOWN_KEYS = REQUIRED_KEYS | {"SUPPORTED_DATASETS", "SOURCE_REF",
                              "INTEGRATION", "IMAGE"}
INTEGRATIONS = {"upstream", "reimplementation"}
# Set by the orchestrator; a method that declares them has them overwritten.
RESERVED = {"DATA", "EXP", "MONITOR_INTERVAL"}

VALID_RESULT_TYPES = {
    "NODE_ANOMALY_SCORES", "EDGE_ANOMALY_SCORES", "GRAPH_ANOMALY_SCORES",
    "TEMPORAL_NODE_ANOMALY_SCORES", "TEMPORAL_EDGE_ANOMALY_SCORES",
    "TEMPORAL_GRAPH_ANOMALY_SCORES",
    "NODE_STREAM_ANOMALY_SCORES", "EDGE_STREAM_ANOMALY_SCORES",
    "GRAPH_STREAM_ANOMALY_SCORES",
}


def parse_env(path: pathlib.Path) -> dict:
    """A copy of ``graflag.utils.parse_env_line``, kept in step by a test.

    The two repositories are independent checkouts, so this cannot import the
    orchestrator's parser. It has to behave identically all the same: these
    tests decide whether a value is acceptable, and a reader that disagreed
    with the one graflag uses would approve a `.env` the cluster then read
    differently -- which is the whole failure mode, one step earlier.

    ``EnvParserAgreement`` below compares the two whenever both checkouts are
    present, so the duplication is checked rather than merely noted.
    """
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[len("export"):].lstrip()
            if "=" not in line:
                continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # A quoted value ends at its closing quote; what follows is a comment.
        if value[:1] in ("'", '"'):
            closing = value.find(value[0], 1)
            if closing != -1:
                out[key] = value[1:closing]
                continue
        # Otherwise an inline comment starts at whitespace followed by '#',
        # so a URL fragment survives.
        for i, ch in enumerate(value):
            if ch == "#" and i > 0 and value[i - 1] in (" ", "\t"):
                value = value[:i]
                break
        out[key] = value.strip()
    return out


def method_dirs():
    return sorted(p for p in METHODS.iterdir() if p.is_dir())


def dockerfile_path(d: pathlib.Path) -> pathlib.Path:
    """The Dockerfile a method actually builds.

    Its own, unless its .env declares IMAGE=, in which case the build uses
    images/<IMAGE>/Dockerfile and the method directory holds no Dockerfile at
    all -- the seventeen bond methods share one. Every check below has to
    follow the same resolution graflag does, or seventeen methods would be
    exempt from the whole file by virtue of the deduplication.
    """
    shared = parse_env(d / ".env").get("IMAGE", "") if (d / ".env").is_file() else ""
    return IMAGES / shared / "Dockerfile" if shared else d / "Dockerfile"


def dockerfile_text(d: pathlib.Path) -> str:
    """A method's Dockerfile with comments dropped and continuations joined.

    Both matter. methods/example/ carries its clone stanza commented out as a
    template, so a naive grep finds a clone in a method that has none; and
    every real clone stanza is a multi-line `&&` chain, so a line-by-line read
    sees the clone and the checkout as unrelated steps.
    """
    path = dockerfile_path(d)
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace").replace("\\\n", " ")
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def clones(d: pathlib.Path) -> bool:
    """Does this method's image fetch an upstream repository?"""
    return bool(re.search(r"\bgit\s+clone\b", dockerfile_text(d)))


CLONING_METHODS = frozenset(d.name for d in method_dirs() if clones(d))


def fetches_upstream(d: pathlib.Path) -> bool:
    """Does this image bring in the authors' own code?

    Two shapes, not one. Seven methods `git clone` SOURCE_CODE; the 17 bond
    methods never clone -- they install the authors' package instead
    (`pip install git+https://.../pygod.git@<sha>`) and select a detector from
    it at run time. Both are upstream code running under the method's name, so
    both count here. Whether the fetch is pinned is a separate question, and
    test_no_dependency_is_installed_from_a_floating_git_ref owns it.
    """
    return clones(d) or "git+https://" in dockerfile_text(d)


class EnvParserAgreement(unittest.TestCase):
    """`parse_env` above against the parser graflag actually uses.

    Only runs in the workspace layout where both checkouts sit side by side --
    the same condition `graflag_data`'s core integration test has. Loading it
    by path rather than importing it keeps `graflag` out of this repository's
    requirements: it is a check on a copy, not a dependency.
    """

    LINES = [
        "METHOD_NAME=taddy",
        "export COMMAND=python3 train_graflag.py",
        "SSH_PORT=22  # default",
        'SUPPORTED_DATASETS="uci, btc_alpha"',
        'DESCRIPTION="TADDY"  # the paper\'s name',
        "SOURCE_CODE=https://docs.pygod.org/x.html#pygod.detector.CoLA",
        "_HIDDEN_DIMS=64 128",
        "_USE_MEMORY=",
        "# a comment",
        "",
        "no_equals_here",
    ]

    def setUp(self):
        import importlib.util
        path = ROOT.parent / "graflag" / "graflag" / "utils.py"
        if not path.is_file():
            self.skipTest("the graflag checkout is not beside this one")
        spec = importlib.util.spec_from_file_location("_graflag_utils", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.reference = module.parse_env_line

    def test_the_copy_agrees_line_by_line(self):
        for line in self.LINES:
            with self.subTest(line=line):
                path = pathlib.Path(self.enterContext(
                    __import__("tempfile").TemporaryDirectory())) / ".env"
                path.write_text(line + "\n")
                mine = parse_env(path)
                theirs = self.reference(line)
                self.assertEqual(mine, {} if theirs is None
                                 else {theirs[0]: theirs[1]})

    def test_every_real_env_reads_the_same_both_ways(self):
        """The copies agree on the lines that exist, not just on examples."""
        for d in method_dirs():
            env = d / ".env"
            if not env.is_file():
                continue
            theirs = {}
            for line in env.read_text(errors="replace").splitlines():
                parsed = self.reference(line)
                if parsed:
                    theirs[parsed[0]] = parsed[1]
            with self.subTest(method=d.name):
                self.assertEqual(parse_env(env), theirs)


class MethodLayout(unittest.TestCase):
    def test_every_method_has_env_and_dockerfile(self):
        for d in method_dirs():
            with self.subTest(method=d.name):
                self.assertTrue((d / ".env").is_file(), "missing .env")
                # Its own, or the shared one its .env names -- a method
                # declaring IMAGE= has no Dockerfile in its directory, and a
                # method naming an image that is not there has nothing to
                # build, which is the failure worth catching here.
                self.assertTrue(dockerfile_path(d).is_file(),
                                f"no Dockerfile at {dockerfile_path(d)}")

    def test_every_method_has_a_readme(self):
        """The .env and the Dockerfile say what a method runs, never what it
        does differently from the paper it cites. dynwalk was the last method
        without one, and its README is where `SOURCE_CODE` pointing at a
        repository the image never clones is written down."""
        missing = sorted(d.name for d in method_dirs()
                         if not (d / "README.md").is_file())
        self.assertEqual(missing, [], f"methods with no README.md: {missing}")

    def test_methods_exist_at_all(self):
        self.assertGreater(len(method_dirs()), 0)

    def test_every_method_records_whether_it_was_verified(self):
        """"Integrated" has to mean something a reader can check.

        A method's README carries a `## Verification` section -- what each
        gate gave, or why it could not run, or that it has not been run -- or
        `VERIFICATION.md` has a row for it. Three methods had neither: one had
        only ever failed on the cluster and two had never been run, and
        nothing in the repository said so.
        """
        matrix = (ROOT / "VERIFICATION.md").read_text()
        in_matrix = set(re.findall(r"^\| `([a-z0-9_]+)` \|", matrix, re.M))
        missing = sorted(
            d.name for d in method_dirs()
            if d.name not in in_matrix and not re.search(
                r"^## Verification\b",
                (d / "README.md").read_text(errors="replace"), re.M))
        self.assertEqual(missing, [], "no recorded verification status: "
                         "add a `## Verification` section to their README")


class EnvFiles(unittest.TestCase):
    def test_required_keys_present(self):
        for d in method_dirs():
            with self.subTest(method=d.name):
                env = parse_env(d / ".env")
                self.assertEqual(REQUIRED_KEYS - set(env), set())

    def test_method_name_matches_directory(self):
        """run() and sync() both lowercase it; a mismatch sends sync somewhere
        the folder name does not predict."""
        for d in method_dirs():
            with self.subTest(method=d.name):
                self.assertEqual(parse_env(d / ".env").get("METHOD_NAME"), d.name)

    def test_no_reserved_variables_declared(self):
        """DATA/EXP/MONITOR_INTERVAL are set by the orchestrator."""
        for d in method_dirs():
            with self.subTest(method=d.name):
                self.assertEqual(RESERVED & set(parse_env(d / ".env")), set())

    def test_parameters_are_uppercase_after_the_underscore(self):
        """The runner lowercases them into CLI flags; mixed case is a trap."""
        for d in method_dirs():
            env = parse_env(d / ".env")
            for key in env:
                if key.startswith("_"):
                    with self.subTest(method=d.name, key=key):
                        self.assertEqual(key, key.upper())

    def test_no_parameter_is_declared_with_an_empty_value(self):
        """An empty value means two different things, so it means neither.

        With ``--pass-env-args`` the runner turns ``_USE_MEMORY=`` into a bare
        ``--use_memory``, which argparse reads as True. ``params()`` reads the
        same empty string as False. methods/gady spelled its flag that way and
        was read the first way, so migrating it to ``params()`` would have
        turned memory off without a word. Spelling the value out -- ``true`` --
        makes both paths agree and makes it readable in service_config.json.
        """
        for d in method_dirs():
            for key, value in parse_env(d / ".env").items():
                if key.startswith("_"):
                    with self.subTest(method=d.name, key=key):
                        self.assertNotEqual(
                            value, "",
                            f"{key} is empty; write the value out "
                            f"(a boolean flag is `true` or `false`)")

    def test_no_unknown_non_parameter_keys(self):
        for d in method_dirs():
            env = parse_env(d / ".env")
            unknown = {k for k in env if not k.startswith("_")} - KNOWN_KEYS
            with self.subTest(method=d.name):
                self.assertEqual(unknown, set())


class SupportedDatasets(unittest.TestCase):
    """The runner refuses to start when nothing matches, so a pattern that
    cannot match any dataset makes the method unrunnable."""

    def available_datasets(self):
        if not DATASETS.is_dir():
            return []
        return sorted(p.name for p in DATASETS.iterdir() if p.is_dir())

    def test_every_pattern_matches_a_known_dataset(self):
        names = self.available_datasets()
        if not names:
            self.skipTest("no datasets/ directory")

        for d in method_dirs():
            patterns = parse_env(d / ".env").get("SUPPORTED_DATASETS", "")
            if not patterns or d.name == "example":
                continue
            for pattern in (p.strip() for p in patterns.split(",") if p.strip()):
                with self.subTest(method=d.name, pattern=pattern):
                    self.assertTrue(
                        any(fnmatch.fnmatchcase(n, pattern) for n in names),
                        f"{pattern!r} matches no dataset directory",
                    )


class Dockerfiles(unittest.TestCase):
    def test_cmd_matches_the_declared_pattern(self):
        """--pass-env-args belongs to methods that run their own script.

        It rewrites COMMAND into `... --batch_size 128`, so putting it on a
        Pattern B method would hand graflag_bond.train CLI flags it never
        parses -- graflag_bond reads the `_FOO` variables itself.

        The converse is not required: `_FOO` reaches the container either way
        (the runner only rewrites the command, runner.py:112), so a script
        that reads its parameters with graflag_runner.params() does not need
        the flag. It is needed when the command parses arguments -- typically
        an upstream argparse entry point that cannot be changed.
        """
        for d in method_dirs():
            df = dockerfile_text(d)
            cmd = re.search(r"^CMD\s+(.+)$", df, re.M)
            with self.subTest(method=d.name):
                self.assertIsNotNone(cmd, "no CMD")
                line = cmd.group(1)
                self.assertIn("graflag_runner", line)
                if "--pass-env-args" not in line:
                    continue
                self.assertTrue(
                    (d / "train_graflag.py").exists(),
                    "--pass-env-args but no train_graflag.py to receive them",
                )
                self.assertNotIn(
                    "graflag_bond", parse_env(d / ".env").get("COMMAND", ""),
                    "graflag_bond.train does not parse CLI arguments",
                )

    def test_copy_paths_are_rooted_at_the_build_context(self):
        """The context is SHARED_DIR, so a COPY names its own directory.

        For a method that is methods/<name>/; for a shared image it is
        images/<image>/, and it cannot be a method directory at all -- a build
        serving seventeen methods that copies from one of them is not shared,
        it is one method's image that sixteen others happen to run.
        """
        for d in method_dirs():
            shared = parse_env(d / ".env").get("IMAGE", "")
            root = f"images/{shared}/" if shared else f"methods/{d.name}/"
            for src in re.findall(r"^COPY\s+(?!--)(\S+)", dockerfile_text(d), re.M):
                with self.subTest(method=d.name, copy=src):
                    self.assertTrue(
                        src.startswith(root) or src.startswith("libs/"),
                        f"{src!r} is not under {root} or libs/",
                    )

    def test_copy_sources_exist(self):
        """A COPY of a file that is not there fails on the manager, not here.

        The rooting check above reads the path's prefix and stops, so a
        Dockerfile kept copying `methods/generaldyg/dataset_all.py` for as long
        as anyone left that line in -- the file having been deleted makes no
        difference to a prefix. What it does make a difference to is the build,
        which dies with `file not found in build context` after the context has
        already been uploaded. Deleting a script and forgetting its COPY is the
        ordinary way to get there.
        """
        for d in method_dirs():
            for src in re.findall(r"^COPY\s+(?!--)(\S+)", dockerfile_text(d), re.M):
                with self.subTest(method=d.name, copy=src):
                    self.assertTrue(
                        (ROOT / src).exists(),
                        f"{src!r} is copied by {d.name} but does not exist",
                    )

    def test_the_graflag_runner_is_installed(self):
        for d in method_dirs():
            df = dockerfile_text(d)
            with self.subTest(method=d.name):
                self.assertRegex(df, r"graflag[-_]runner")

    def test_bond_methods_install_graflag_bond(self):
        for d in method_dirs():
            env = parse_env(d / ".env")
            if "graflag_bond" not in env.get("COMMAND", ""):
                continue
            df = dockerfile_text(d)
            with self.subTest(method=d.name):
                self.assertRegex(df, r"graflag[-_]bond")


def string_dicts(tree):
    """Module-level `NAME = {...: "str"}` tables, as {name: {values}}."""
    tables = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        values = {v.value for v in node.value.values
                  if isinstance(v, ast.Constant) and isinstance(v.value, str)}
        if not values or len(values) != len(node.value.values):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                tables[target.id] = values
    return tables


def result_types_published(tree):
    """(types a script can pass to save_scores, expressions we could not read).

    A literal resolves to itself; `TABLE[key]` resolves to every value of a
    module-level table of strings. Anything else is reported as unresolved so
    it fails the test instead of silently contributing nothing.
    """
    tables = string_dicts(tree)
    used, unresolved = set(), []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "save_scores"):
            continue
        for kw in node.keywords:
            if kw.arg != "result_type":
                continue
            value = kw.value
            if isinstance(value, ast.Constant):
                used.add(value.value)
            elif (isinstance(value, ast.Subscript)
                  and isinstance(value.value, ast.Name)
                  and value.value.id in tables):
                used |= tables[value.value.id]
            else:
                unresolved.append(ast.dump(value))
    return used, unresolved


class IntegrationScripts(unittest.TestCase):
    def scripts(self):
        return [(d, d / "train_graflag.py") for d in method_dirs()
                if (d / "train_graflag.py").is_file()]

    def test_scripts_parse(self):
        for d, f in self.scripts():
            with self.subTest(method=d.name):
                ast.parse(f.read_text(errors="replace"))

    def test_every_script_declares_what_it_scored(self):
        """`scored_split` and `scored_samples` in the summary, always.

        They are what `graflag verify` reads to tell a test-split AUC from a
        whole-stream one, and the scored count from the published count.
        Missing, it can only warn that it cannot tell -- which it did for 24
        of the 33 methods, the annotated template included, while that
        template's own comment said the scores must come from the test split
        and its code scored every edge. `graflag_bond` writes the summary for
        the seventeen PyGOD methods, so it is held to the same rule.
        """
        sources = [(d.name, f) for d, f in self.scripts()]
        sources.append(("graflag_bond", ROOT / "libs" / "graflag_bond" / "train.py"))
        for name, f in sources:
            tree = ast.parse(f.read_text(errors="replace"))
            keys = {
                key.value
                for call in ast.walk(tree)
                if isinstance(call, ast.Call)
                and getattr(call.func, "attr", None) == "add_metadata"
                for node in ast.walk(call)
                if isinstance(node, ast.Dict)
                for key in node.keys
                if isinstance(key, ast.Constant)
            }
            with self.subTest(method=name):
                self.assertTrue({"scored_split", "scored_samples"} <= keys,
                                "add_metadata(summary=...) must record "
                                "scored_split and scored_samples")

    def test_result_types_are_valid(self):
        """Every result type a script can publish has to be a real one.

        `graflag_runner` accepts the string as given and the evaluator picks
        its metric registry from it, so a typo here is not caught until the
        experiment has already run and `graflag evaluate` finds no calculator.

        The type is usually a literal. `anograph` serves four upstream
        algorithms, two of which score time windows and two of which score
        edges, so it selects one through a module-level table -- which is
        resolved here rather than waved through, because "not a literal" must
        not become the hole every wrong result type fits through.
        """
        for d, f in self.scripts():
            tree = ast.parse(f.read_text(errors="replace"))
            used, unresolved = result_types_published(tree)
            with self.subTest(method=d.name):
                self.assertEqual(
                    unresolved, [],
                    f"save_scores result_type is not statically resolvable: "
                    f"{unresolved}")
                self.assertTrue(used, "save_scores declares no result_type")
                self.assertEqual(used - VALID_RESULT_TYPES, set())

    def test_a_problem_is_never_signalled_by_a_bare_print(self):
        """A `print("Warning: ...")` is a problem the run then ignores.

        streamspot's parse_streamspot_output printed
        `Warning: Expected 600 scores, got N` and returned the short list
        anyway. ResultWriter.save_scores does not check that `scores` and
        `ground_truth` are the same length, so the mismatch was published and
        only surfaced much later -- inside sklearn ("Found input variables
        with inconsistent numbers of samples") or on an IndexError in the
        scenario filter, whichever came first. Either way the message naming
        the real cause had scrolled past hundreds of lines earlier.

        A script has two honest options: raise, or say it through
        graflag_runner.warning(), which carries the [WARN] prefix this project
        writes and goes to the logger rather than into method_output.txt as
        indistinguishable stdout. This test only forbids the third.
        """
        pattern = re.compile(r"^(warning|error|fatal)\b", re.I)

        def announces_a_problem(node):
            """A print() whose first argument starts with Warning:/Error:."""
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "print"
                    and node.args):
                return False
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return bool(pattern.match(first.value))
            # An f-string: the literal head is its first JoinedStr part.
            if isinstance(first, ast.JoinedStr) and first.values:
                head = first.values[0]
                return (isinstance(head, ast.Constant)
                        and isinstance(head.value, str)
                        and bool(pattern.match(head.value)))
            return False

        for d, f in self.scripts():
            tree = ast.parse(f.read_text(errors="replace"))
            offenders = [n.lineno for n in ast.walk(tree)
                         if announces_a_problem(n)]
            with self.subTest(method=d.name):
                self.assertEqual(
                    offenders, [],
                    f"{f.name} announces a problem with a bare print() at "
                    f"line(s) {offenders}; raise, or use "
                    f"graflag_runner.warning() for the [WARN] prefix")

    def test_a_missing_input_is_never_skipped(self):
        """A file that should be there and is not must stop the run.

        gady read one savepoint of upstream's positional encodings per batch
        and wrapped the read in

            try:
                V, R = torch.load(path)
            except FileNotFoundError:
                continue

        Those savepoints are written by upstream's preproc_new.py, which
        data_loader.py runs before training. If that step had not run, every
        batch of every epoch took the `continue`: the loop completed, the
        method reported success, and it had trained on nothing. Nothing in the
        output said so -- an empty epoch and a finished epoch look identical
        from outside.

        The distinction this draws is between an input and a token. Skipping a
        non-numeric token while parsing a line is a parser doing its job, and
        anograph and streamspot both do it. Skipping a *file* is a decision
        that the run can proceed without one of its inputs, which is never a
        decision a per-item handler is in a position to make.

        So a handler for a filesystem error may raise, or announce the problem
        through graflag_runner.warning()/error() -- what it may not do is
        return to the loop with nothing said.
        """
        FS_ERRORS = {"FileNotFoundError", "IOError", "OSError", "EnvironmentError"}
        SPEAKS = {"raise", "warning", "error", "fatal"}

        def caught(handler):
            t = handler.type
            if t is None:
                return {"<bare>"}
            if isinstance(t, ast.Name):
                return {t.id}
            if isinstance(t, ast.Tuple):
                return {e.id for e in t.elts if isinstance(e, ast.Name)}
            return set()

        def says_something(handler):
            for node in ast.walk(handler):
                if isinstance(node, ast.Raise):
                    return True
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
                    if name.lower() in SPEAKS:
                        return True
            return False

        for d, f in self.scripts():
            tree = ast.parse(f.read_text(errors="replace"))
            offenders = [
                h.lineno for h in ast.walk(tree)
                if isinstance(h, ast.ExceptHandler)
                and (caught(h) & FS_ERRORS or caught(h) == {"<bare>"})
                and not says_something(h)
            ]
            with self.subTest(method=d.name):
                self.assertEqual(
                    offenders, [],
                    f"{f.name} swallows a missing-file error at line(s) "
                    f"{offenders} without raising or announcing it; a run "
                    f"that proceeds without one of its inputs cannot be told "
                    f"from one that had them")

    def test_scripts_write_a_result(self):
        for d, f in self.scripts():
            src = f.read_text(errors="replace")
            with self.subTest(method=d.name):
                self.assertIn("save_scores", src)
                self.assertIn("finalize()", src)
                self.assertIn("ground_truth", src)


class SdkContract(unittest.TestCase):
    """A script may only reach for members the SDK actually has.

    `paths()` returns an ExperimentPaths, and a script that asks it for an
    attribute it does not carry -- `run.experiment_dir` for `run.exp`, say --
    parses, imports and passes every other check here, then raises
    AttributeError minutes into a cluster run. Resolving the names against the
    real dataclass is what turns that into a test failure.
    """

    def sdk_members(self, cls_name):
        sys.path.insert(0, str(ROOT / "libs"))
        try:
            import graflag_runner.method as method
        finally:
            sys.path.pop(0)
        cls = getattr(method, cls_name)
        # dir() alone misses the dataclass fields: `data` and `exp` have no
        # defaults, so they exist on instances only.
        return set(dir(cls)) | {f.name for f in dataclasses.fields(cls)}

    def test_experiment_paths_attributes_exist(self):
        members = self.sdk_members("ExperimentPaths")
        for d in method_dirs():
            script = d / "train_graflag.py"
            if not script.is_file():
                continue
            tree = ast.parse(script.read_text(errors="replace"))

            # The name paths() was bound to, usually `run`.
            holders = {
                target.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "") == "paths"
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            used = {
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in holders
            }
            with self.subTest(method=d.name):
                self.assertEqual(
                    used - members, set(),
                    f"ExperimentPaths has no such attribute; it has "
                    f"{sorted(m for m in members if not m.startswith('_'))}")


class SkillReference(unittest.TestCase):
    """The skill's SDK reference may only name what graflag_runner has.

    An agent follows `reference/sdk.md` literally; a function it names that the
    library does not export is an ImportError minutes into a cluster run.
    Checked against the source, so nothing is imported.
    """

    SDK_MD = ROOT / ".claude" / "skills" / "method-integration" / "reference" / "sdk.md"
    LIB = ROOT / "libs" / "graflag_runner"

    def exported(self):
        tree = ast.parse((self.LIB / "__init__.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "__all__" for t in node.targets):
                return set(ast.literal_eval(node.value))
        self.fail("graflag_runner has no __all__")

    def writer_methods(self):
        tree = ast.parse((self.LIB / "results.py").read_text())
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == "ResultWriter")
        return {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}

    def test_every_imported_name_is_exported(self):
        text = self.SDK_MD.read_text()
        names = set()
        for group, line in re.findall(
                r"from graflag_runner import\s*(?:\(([^)]*)\)|([^\n`]+))", text):
            names |= {n.strip() for n in (group or line).split(",") if n.strip()}
        self.assertGreater(len(names), 5, "found no imports to check")
        self.assertEqual(names - self.exported(), set())

    def test_every_call_in_the_table_is_exported(self):
        text = self.SDK_MD.read_text()
        calls = set()
        for row in re.findall(r"^\|[^|\n]+\|([^|\n]+)\|[^|\n]+\|$", text, re.M):
            calls |= set(re.findall(r"`([A-Za-z_]+)\(", row))
        self.assertIn("params", calls, "the table was not parsed")
        self.assertEqual(calls - self.exported(), set())

    def test_every_writer_method_exists(self):
        used = set(re.findall(r"writer\.([a-z_]+)\(", self.SDK_MD.read_text()))
        self.assertIn("finalize", used)
        self.assertEqual(used - self.writer_methods(), set())


class GpuConvention(unittest.TestCase):
    """`_GPU=-1` means CPU, for every method and not just the PyGOD ones.

    `graflag run --no-gpu` drops the Swarm resource reservation and switches
    the method's own `_GPU` to -1, because otherwise a service scheduled
    without a GPU is still told to use cuda:0. That switch is only safe while
    every method reads -1 as "CPU": three that built a device string by hand
    turned it into `torch.device("cuda:-1")`, which raises
    "Invalid device string", and the orchestrator was narrowed to bond
    methods to work around it. This is the check that let it be widened
    again -- see graflag/tests/test_reproducibility.py::NoGpuReachesTheMethod.
    """

    def scripts(self):
        return [(d, d / "train_graflag.py") for d in method_dirs()
                if (d / "train_graflag.py").is_file()]

    @staticmethod
    def _parents(tree):
        parent = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
        return parent

    @staticmethod
    def _cuda_strings(tree):
        """Every node that builds a "cuda:<index>" string."""
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                head = node.values[0] if node.values else None
                if isinstance(head, ast.Constant) and str(head.value).startswith("cuda:"):
                    found.append(node)
        return found

    @staticmethod
    def _is_nonnegative_test(node):
        return any(
            isinstance(cmp, ast.Compare)
            and any(isinstance(op, (ast.GtE, ast.Gt)) for op in cmp.ops)
            and any(isinstance(c, ast.Constant) and c.value == 0 for c in cmp.comparators)
            for cmp in ast.walk(node)
        )

    def _uses_sdk_device(self, tree):
        return any(
            isinstance(n, ast.ImportFrom) and n.module == "graflag_runner"
            and any(a.name == "device" for a in n.names)
            for n in ast.walk(tree)
        )

    def test_a_hand_built_cuda_string_is_guarded_on_a_nonnegative_index(self):
        for d, f in self.scripts():
            tree = ast.parse(f.read_text(errors="replace"))
            if self._uses_sdk_device(tree):
                continue                      # device() owns the convention
            parents = self._parents(tree)
            for node in self._cuda_strings(tree):
                with self.subTest(method=d.name, line=node.lineno):
                    guarded, cur = False, node
                    while cur in parents and not guarded:
                        cur = parents[cur]
                        if isinstance(cur, ast.IfExp) and self._is_nonnegative_test(cur.test):
                            guarded = True
                        elif isinstance(cur, ast.If) and self._is_nonnegative_test(cur.test):
                            guarded = True
                        elif isinstance(cur, ast.Assign) and self._is_nonnegative_test(cur.value):
                            guarded = True
                        elif isinstance(cur, (ast.FunctionDef, ast.Module)):
                            break
                    if not guarded:
                        # The guard may be a `_use_cuda = ... >= 0` computed
                        # earlier in the same function; accept that too.
                        fn = node
                        while fn in parents and not isinstance(fn, ast.FunctionDef):
                            fn = parents[fn]
                        guarded = isinstance(fn, ast.FunctionDef) and self._is_nonnegative_test(fn)
                    self.assertTrue(
                        guarded,
                        f"{d.name}: builds a cuda: string with no `>= 0` guard, so "
                        f"--no-gpu turns _GPU=-1 into the invalid 'cuda:-1'. Use "
                        f"graflag_runner.device() instead.")

    def test_a_method_that_picks_a_device_declares_gpu(self):
        """Without `_GPU` in the .env there is nothing for --no-gpu to switch,
        so the method takes a GPU on a service scheduled without one."""
        for d, f in self.scripts():
            tree = ast.parse(f.read_text(errors="replace"))
            picks = self._uses_sdk_device(tree) or bool(self._cuda_strings(tree))
            if not picks:
                continue
            with self.subTest(method=d.name):
                self.assertIn("_GPU", parse_env(d / ".env"),
                              "selects a device but declares no _GPU")

    def test_a_cpu_only_image_does_not_declare_a_gpu_index(self):
        """An image built on the CPU-only torch wheel can never reach cuda.

        dynwalk's Dockerfile installs
        `torch --index-url https://download.pytorch.org/whl/cpu` while its
        .env declared `_GPU=0`. Nothing crashed -- device() finds no GPU and
        returns cpu -- so the mismatch was invisible in the logs, and the
        .env went on advertising a device the container could not use. It
        matters because the reservation is decided elsewhere: `graflag run`
        asks Swarm for an NVIDIA-GPU by default (core.py, `gpu: bool = True`),
        so a run of a CPU-only method holds a card nothing in it can touch.
        The .env is the only place that mismatch is visible, so keep it true.
        """
        for d in method_dirs():
            dockerfile = dockerfile_path(d)
            if not dockerfile.is_file():
                continue
            if "download.pytorch.org/whl/cpu" not in dockerfile.read_text(
                    errors="replace"):
                continue
            declared = parse_env(d / ".env").get("_GPU", "-1").strip()
            with self.subTest(method=d.name):
                self.assertEqual(
                    declared, "-1",
                    f"installs the CPU-only torch wheel but declares "
                    f"_GPU={declared}")


class ClonedSourceIsUsed(unittest.TestCase):
    """A `git clone` that nothing reads is a claim the results do not support.

    addgraph cloned https://github.com/Ljiajie/Addgraph and its script opened
    with sys.path.insert(0, '/app/src/UCI_D_Addgraph/framwork') -- a path no
    import ever resolved against. The models are reimplemented in
    train_graflag.py, so the download was pure weight, and its presence said
    the numbers came from upstream's code when they did not.

    A clone counts as read if the script imports a module nothing else
    supplies, or if a later Dockerfile step names the directory. Both are
    needed: gady and strgnn import from theirs, while anograph and streamspot
    compile theirs and never import it.
    """

    # Distribution name -> the name you import it under.
    ALIASES = {"scikit-learn": "sklearn", "pyyaml": "yaml", "pillow": "PIL"}

    def provided(self, dockerfile):
        """Top-level module names the image supplies without any clone."""
        names = set(sys.stdlib_module_names)
        for line in dockerfile.splitlines():
            if not re.search(r"\bpip3?\s+install\b", line):
                continue
            for token in line.split():
                token = token.strip("\"'")
                if (token.startswith("-") or "/" in token
                        or token in ("pip", "pip3", "install", "RUN", "&&")):
                    continue
                dist = re.split(r"[=<>~\[]", token)[0].lower()
                if dist:
                    names.add(self.ALIASES.get(dist, dist.replace("-", "_")))
        for src in re.findall(r"^COPY\s+(?!--)(\S+)", dockerfile, re.M):
            if src.endswith(".py"):
                names.add(Path(src).stem)
        return names

    def imported(self, script):
        tops = set()
        for node in ast.walk(ast.parse(script.read_text(errors="replace"))):
            if isinstance(node, ast.Import):
                tops |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                tops.add(node.module.split(".")[0])
        return tops

    def test_every_clone_is_referenced_somewhere(self):
        for d in method_dirs():
            dockerfile = dockerfile_text(d)
            script = d / "train_graflag.py"

            supplied = self.provided(dockerfile)
            from_clone = (self.imported(script) - supplied) if script.exists() else set()

            for match in re.finditer(
                    r"git\s+clone\s+(?:--\S+\s+)*(\S+)(?:\s+([\w.\-/]+))?", dockerfile):
                url, target = match.group(1), match.group(2)
                if not target:
                    target = url.rsplit("/", 1)[-1].removesuffix(".git")
                # The pin stanza names the target too (`git -C src checkout
                # --detach ...`, `git -C src apply ...`). Those are part of
                # fetching the clone, not evidence that anything reads it, so
                # they are dropped before looking for a step that does --
                # otherwise every clone would vouch for itself and this test
                # would pass for a method that downloads a repository and
                # ignores it, which is the case it exists to catch.
                rest = re.sub(rf"git\s+-C\s+{re.escape(target)}\s+"
                              r"(?:checkout|apply)\b[^\n]*", "",
                              dockerfile[match.end():])
                built_from = re.search(rf"(?:^|[\s/=]){re.escape(target)}(?:[\s/]|$)",
                                       rest, re.M)
                with self.subTest(method=d.name, clone=target):
                    self.assertTrue(
                        from_clone or built_from,
                        f"{d.name} clones {url} into {target!r}: no import "
                        f"resolves against it and no later step builds from "
                        f"it, so nothing in the image reads it",
                    )


class PinnedUpstreams(unittest.TestCase):
    """A clone with no commit is not a build, it is a bet on a stranger.

    Every `git clone` under methods/ used to float on the default branch.
    taddy's build broke with a 404 and gady's with two TypeErrors from
    constructor signatures that had moved; all three were one defect -- an
    image whose contents were decided by the calendar. The quieter cost is
    that rebuilding an experiment months later produced a different method
    wearing the same name, and nothing in the record said so.

    SOURCE_CODE and SOURCE_REF are declared in the .env and reach the build as
    `--build-arg` (graflag.docker_ops.BUILD_ARG_KEYS), so the Dockerfile
    repeats neither and the two cannot drift apart.
    """

    SHA = r"^[0-9a-f]{40}$"

    def test_a_method_that_clones_declares_a_source_ref(self):
        for d in method_dirs():
            if not clones(d):
                continue
            with self.subTest(method=d.name):
                self.assertIn(
                    "SOURCE_REF", parse_env(d / ".env"),
                    f"{d.name} clones an upstream but pins no commit")

    def test_a_declared_source_ref_is_a_full_sha(self):
        """Not a tag, not a branch, not an abbreviation.

        A tag can be moved and a branch is moved by definition; a short sha
        stops naming one object as the repository grows. Only the full 40-char
        name says the same thing forever, which is the entire point of writing
        it down.
        """
        for d in method_dirs():
            ref = parse_env(d / ".env").get("SOURCE_REF")
            if ref is None:
                continue
            with self.subTest(method=d.name):
                self.assertRegex(
                    ref, self.SHA,
                    f"{d.name} pins SOURCE_REF={ref!r}, which is not a full "
                    f"40-character commit sha")

    def test_a_source_ref_belongs_to_a_method_that_clones(self):
        """A pin nobody reads is worse than no pin: it reads as a guarantee."""
        for d in method_dirs():
            if "SOURCE_REF" not in parse_env(d / ".env"):
                continue
            with self.subTest(method=d.name):
                self.assertTrue(
                    clones(d),
                    f"{d.name} declares SOURCE_REF but its Dockerfile clones "
                    f"nothing, so the pin describes code the image never gets")

    def test_every_clone_lands_on_the_pinned_commit(self):
        """The checkout has to be in the same RUN as the clone.

        Not style: chained with `&&`, the clone cannot succeed without the
        checkout. Split across two RUNs, deleting the second leaves a build
        that still works and is no longer pinned -- and the layer cache would
        happily keep serving the floating clone.
        """
        for d in method_dirs():
            for line in dockerfile_text(d).splitlines():
                if not re.search(r"\bgit\s+clone\b", line):
                    continue
                with self.subTest(method=d.name):
                    self.assertRegex(
                        line,
                        r"git\s+-C\s+\S+\s+checkout\s+--detach\s+\$\{SOURCE_REF\}",
                        f"{d.name}: this RUN clones without checking out "
                        f"${{SOURCE_REF}}, so the image is built from whatever "
                        f"the default branch held that day:\n  {line.strip()}")

    def test_an_unset_source_ref_stops_the_build(self):
        """`git checkout --detach` with no argument exits 0.

        So an unset `ARG SOURCE_REF` -- a `docker build` run by hand, a typo in
        the key, an orchestrator that stops forwarding it -- would detach at the
        default branch and the pin would be gone with nothing printed. The
        `test -n` is what turns that into a failed build instead.
        """
        for d in method_dirs():
            for line in dockerfile_text(d).splitlines():
                if not re.search(r"\bgit\s+clone\b", line):
                    continue
                with self.subTest(method=d.name):
                    self.assertIn(
                        'test -n "${SOURCE_REF}"', line,
                        f"{d.name}: nothing here rejects an empty SOURCE_REF, "
                        f"and an empty one checks out silently")

    def test_the_build_args_are_declared_in_the_stage_that_clones(self):
        """An ARG declared in another stage expands to the empty string.

        Docker scopes ARG per stage, so in a multi-stage Dockerfile -- which
        streamspot is -- an `ARG SOURCE_REF` above the wrong FROM leaves
        ${SOURCE_REF} empty in the builder with no warning at all.
        """
        for d in method_dirs():
            if not clones(d):
                continue
            stage = []
            for line in dockerfile_text(d).splitlines():
                if re.match(r"\s*FROM\b", line, re.I):
                    stage = []
                stage.append(line)
                if not re.search(r"\bgit\s+clone\b", line):
                    continue
                declared = {m.group(1) for m in
                            (re.match(r"\s*ARG\s+(\w+)", l) for l in stage) if m}
                with self.subTest(method=d.name):
                    self.assertLessEqual(
                        {"SOURCE_CODE", "SOURCE_REF"}, declared,
                        f"{d.name}: the stage that clones declares "
                        f"{sorted(declared)}; both SOURCE_CODE and SOURCE_REF "
                        f"must be re-declared after its FROM")

    def test_the_upstream_url_is_not_repeated_in_the_dockerfile(self):
        """One name for the repository, and it is the one `list methods` shows.

        dynwalk's .env advertised NetWalk while its image cloned nothing, and
        gady, slade and generaldyg each carried the URL twice -- once in the
        .env, once in an `ENV SOURCE_CODE=` the .env could not see. Two copies
        is one copy too many to keep true.
        """
        for d in method_dirs():
            if not clones(d):
                continue
            with self.subTest(method=d.name):
                self.assertNotRegex(
                    dockerfile_text(d), r"(?:git\s+clone\s+|ENV\s+SOURCE_CODE=)https://",
                    f"{d.name} hardcodes its upstream URL; clone "
                    f"${{SOURCE_CODE}} and let the .env be the only copy")

    def test_no_dependency_is_installed_from_a_floating_git_ref(self):
        """The same defect one layer down.

        All 17 bond methods installed PyGOD with `--upgrade git+...pygod.git`
        and no ref, so the detector signatures their .env files were written
        against could change between two rebuilds of the same method. When they
        do, get_all_parameters quietly drops the parameters the constructor no
        longer takes, and the run still reports success.
        """
        for d in method_dirs():
            for m in re.finditer(r"git\+https://\S+", dockerfile_text(d)):
                url = m.group(0).rstrip("\\").rstrip()
                with self.subTest(method=d.name, url=url):
                    self.assertRegex(
                        url.partition("@")[2].split("#")[0], self.SHA,
                        f"{d.name} installs {url} without a pinned commit")

    def test_every_in_place_rewrite_of_cloned_source_is_verified(self):
        """`sed -i` exits 0 when it matches nothing.

        streamspot is the case that bites: its rewrite dropped
        `-march=native`, and a silent no-op there ships a binary tuned for the
        manager's CPU that SIGILLs on an older worker -- a crash that reads as
        a method bug. So a RUN that edits cloned source in place has to check
        itself, either with `git apply` (which refuses a patch whose context
        moved) or with greps bracketing the sed.
        """
        for d in method_dirs():
            for line in dockerfile_text(d).splitlines():
                if not re.search(r"\bsed\s+-i\b", line):
                    continue
                with self.subTest(method=d.name):
                    self.assertRegex(
                        line, r"\bgrep\b",
                        f"{d.name} rewrites source in place with nothing "
                        f"asserting the rewrite happened:\n  {line.strip()}")

    def test_a_patch_directory_is_applied_and_an_applied_one_exists(self):
        """patches/ and the `git apply` that uses it have to agree.

        A patch file nobody applies is a fix that is not in the image; an apply
        of a directory that is not there fails the build at the last moment,
        after every dependency layer has been rebuilt.
        """
        for d in method_dirs():
            text = dockerfile_text(d)
            applies = bool(re.search(r"git\s+(?:-C\s+\S+\s+)?apply\b", text))
            have = (d / "patches").is_dir() and any((d / "patches").glob("*.patch"))
            with self.subTest(method=d.name):
                self.assertEqual(
                    applies, have,
                    f"{d.name}: Dockerfile applies patches={applies}, "
                    f"patches/*.patch present={have}")
            if not have:
                continue
            with self.subTest(method=d.name, copies="patches"):
                self.assertIn(f"COPY methods/{d.name}/patches/", text)





class GraflagLibraries(unittest.TestCase):
    """How a method image gets graflag_runner, and whether that is a choice.

    Every Dockerfile used to hardcode `pip install graflag-runner`, so what a
    container ran was whatever was last released to PyPI. `graflag sync --lib`
    pushes the libraries to the share and reported success while changing
    nothing that ran -- and the only way to test a library change was to edit
    the Dockerfiles on the manager by hand. That happened: eight of them, in
    three different spellings, and the manager held the only copy.

    `ARG GRAFLAG_LIBS` makes the choice explicit and identical everywhere.
    graflag passes it from its config; these tests keep the Dockerfiles able
    to receive it.
    """

    PYPI_INSTALL = re.compile(r"\bpip3?\s+install\b[^\n]*\bgraflag-\w+")

    def installers(self):
        """Every Dockerfile that installs a GraFlag library, resolved once."""
        seen = {}
        for d in method_dirs():
            path = dockerfile_path(d)
            if path.is_file() and re.search(r"graflag[-_](runner|bond)",
                                            dockerfile_text(d)):
                seen[path] = dockerfile_text(d)
        return seen

    def test_every_method_image_installs_the_libraries(self):
        """A method that installs neither cannot run under the runner."""
        for d in method_dirs():
            with self.subTest(method=d.name):
                self.assertRegex(dockerfile_text(d), r"graflag[-_]runner")

    def test_the_source_of_the_libraries_is_a_build_argument(self):
        """Not a hardcoded `pip install graflag-runner`.

        That line is what made a library change untestable without editing
        the image by hand, and it is the line the manager's copies diverged
        on.
        """
        for path, text in self.installers().items():
            with self.subTest(dockerfile=str(path.relative_to(ROOT))):
                self.assertIn("ARG GRAFLAG_LIBS", text)
                bare = [l for l in text.splitlines()
                        if self.PYPI_INSTALL.search(l) and "case" not in l
                        and "pypi)" not in l]
                self.assertEqual(bare, [], "installs a GraFlag library "
                                           "without honouring GRAFLAG_LIBS")

    def test_both_sources_are_offered(self):
        for path, text in self.installers().items():
            with self.subTest(dockerfile=str(path.relative_to(ROOT))):
                self.assertIn("local)", text)
                self.assertIn("pypi)", text)

    def test_an_unknown_value_stops_the_build(self):
        """The arm that makes a typo loud.

        With an if/else, GRAFLAG_LIBS=locl builds from PyPI and says nothing
        -- an image whose libraries are not the ones asked for, reported as a
        success. That is the same shape as `git checkout --detach` with no
        argument, which this tree spent a phase removing.
        """
        for path, text in self.installers().items():
            with self.subTest(dockerfile=str(path.relative_to(ROOT))):
                self.assertRegex(text, r"\*\)[^\n]*exit 1")

    def test_the_local_source_is_the_copy_that_was_made(self):
        """A path typo here installs nothing and the build still succeeds:
        pip would resolve the name from PyPI instead."""
        for path, text in self.installers().items():
            copied = set(re.findall(r"^COPY\s+libs/(\S+)\s+(\S+)", text, re.M))
            targets = {dst for _, dst in copied}
            for used in re.findall(r"local\)[^\n]*?((?:/tmp/\S+\s*)+);;", text):
                for one in used.split():
                    with self.subTest(dockerfile=str(path.relative_to(ROOT)),
                                      path=one):
                        self.assertIn(one, targets,
                                      f"{one} is installed but never copied")

    def test_the_copied_libraries_are_the_installed_ones(self):
        """A COPY nothing installs is dead weight in the image."""
        for path, text in self.installers().items():
            for src, dst in re.findall(r"^COPY\s+libs/(\S+)\s+(\S+)", text, re.M):
                with self.subTest(dockerfile=str(path.relative_to(ROOT)), lib=src):
                    self.assertTrue((ROOT / "libs" / src).is_dir(),
                                    f"libs/{src} is not in this repository")
                    self.assertIn(dst, text.split("local)", 1)[-1].split(";;", 1)[0],
                                  f"libs/{src} is copied but not installed")


class SharedImages(unittest.TestCase):
    """A method builds its own image unless it says otherwise.

    The seventeen bond Dockerfiles were byte-identical, and the only thing
    forcing seventeen builds of the same ~9 GB image was the image name being
    derived from the method name. That is not a tidiness problem: 17 x 9 GB
    against the cluster's free disk is why building the whole method tree was
    impossible, so the deduplication is what makes running all 27 methods a
    thing that can happen at all.

    IMAGE= in the .env opts in. The methods stay distinct at run time --
    METHOD_NAME is set per service and graflag_bond.train reads it -- so the
    image holds nothing method-specific and these tests keep it that way.
    """

    NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

    def shared(self):
        return [(d, parse_env(d / ".env").get("IMAGE", "")) for d in method_dirs()]

    def test_a_declared_image_exists(self):
        for d, image in self.shared():
            if not image:
                continue
            with self.subTest(method=d.name, image=image):
                self.assertTrue(
                    (IMAGES / image / "Dockerfile").is_file(),
                    f"{d.name} declares IMAGE={image}, but there is no "
                    f"images/{image}/Dockerfile to build")

    def test_a_declared_image_is_usable_as_a_docker_repository(self):
        """It is both a directory name and half of an image reference.

        Docker rejects an uppercase repository name, and the .env is the only
        place the two spellings are stated once -- so a name that is not legal
        fails at push time, after the build has already run.
        """
        for d, image in self.shared():
            if not image:
                continue
            with self.subTest(method=d.name, image=image):
                self.assertRegex(image, self.NAME)

    def test_a_method_sharing_an_image_keeps_no_dockerfile_of_its_own(self):
        """The copy left behind is the one that drifts.

        It builds nothing -- graflag resolves the path from IMAGE= -- so an
        edit to it silently has no effect, which is exactly how the seventeen
        came to be maintained in parallel in the first place.
        """
        for d, image in self.shared():
            if not image:
                continue
            with self.subTest(method=d.name):
                self.assertFalse(
                    (d / "Dockerfile").is_file(),
                    f"{d.name} declares IMAGE={image} but still has its own "
                    f"Dockerfile, which nothing builds")

    def test_a_method_declaring_no_image_has_its_own(self):
        for d, image in self.shared():
            if image:
                continue
            with self.subTest(method=d.name):
                self.assertTrue((d / "Dockerfile").is_file())

    def test_every_shared_image_is_claimed_by_a_method(self):
        """An image nothing declares is built by nothing and run by nothing."""
        declared = {image for _, image in self.shared() if image}
        present = {p.name for p in IMAGES.iterdir() if p.is_dir()} if IMAGES.is_dir() else set()
        self.assertEqual(sorted(present - declared), [],
                         "images/ holds an image no method declares")

    def test_no_two_methods_build_the_same_dockerfile_separately(self):
        """The forward guard, and the one that would have caught this.

        Seventeen identical files are not a thing anyone decides to write;
        they arrive one copy at a time, each a reasonable local act. A method
        added by copying another's Dockerfile now fails here with the name of
        the file it duplicated, and the fix is IMAGE= rather than an
        eighteenth copy.
        """
        by_content = {}
        for d in method_dirs():
            path = dockerfile_path(d)
            if not path.is_file():
                continue
            by_content.setdefault(path.read_bytes(), set()).add(path)
        for paths in by_content.values():
            if len(paths) > 1:
                names = sorted(str(p.relative_to(ROOT)) for p in paths)
                self.fail(f"identical Dockerfiles that could be one shared "
                          f"image: {names}")


class Provenance(unittest.TestCase):
    """Whose numbers a run produces is a property of the image, not a detail.

    `SOURCE_CODE` was being read as "this is what ran", and twice it was not:
    dynwalk's named NetWalk while its image cloned nothing, and addgraph's
    named a repository the image put on sys.path and never imported. Both
    published an AUC under the paper's name that the paper's code had no part
    in producing. INTEGRATION states which of the two a method is, and these
    tests keep that claim tied to what the Dockerfile actually fetches -- so a
    reimplementation that later grows a clone, or an upstream method whose
    clone is dropped, fails here instead of silently changing what a published
    number means.
    """

    def declared(self):
        return [(d, parse_env(d / ".env").get("INTEGRATION", "")) for d in method_dirs()]

    def test_every_method_declares_where_its_results_come_from(self):
        missing = sorted(d.name for d, value in self.declared() if not value)
        self.assertEqual(missing, [],
                         f"methods that do not declare INTEGRATION: {missing}")

    def test_the_declared_value_is_one_of_the_two(self):
        """A typo is worse than an omission: it reads as an answer.

        Nothing consumes an unrecognised value -- `graflag list methods`
        prints it verbatim -- so `INTEGRATION=partial` would look decided and
        mean nothing.
        """
        for d, value in self.declared():
            if not value:
                continue
            with self.subTest(method=d.name):
                self.assertIn(value, INTEGRATIONS,
                              f"{d.name} declares INTEGRATION={value!r}, "
                              f"which is neither of {sorted(INTEGRATIONS)}")

    def test_upstream_means_the_image_fetches_upstream_code(self):
        for d, value in self.declared():
            if value != "upstream":
                continue
            with self.subTest(method=d.name):
                self.assertTrue(
                    fetches_upstream(d),
                    f"{d.name} claims to run the authors' implementation, but "
                    f"its Dockerfile neither clones nor installs one")

    def test_a_reimplementation_fetches_nothing(self):
        """The claim cuts both ways.

        A method saying the results are this code's while the image clones the
        authors' is the same mislabelling as dynwalk's, pointed the other way,
        and it is the direction a later edit is likely to take: someone adds a
        clone to borrow a data loader and leaves the key alone.
        """
        for d, value in self.declared():
            if value != "reimplementation":
                continue
            with self.subTest(method=d.name):
                self.assertFalse(
                    fetches_upstream(d),
                    f"{d.name} says its results are this repository's, but its "
                    f"image fetches upstream code -- one of the two is wrong")

    def test_a_reimplementation_carries_the_code_it_claims(self):
        """If nothing is fetched and nothing is written here, nothing runs."""
        for d, value in self.declared():
            if value != "reimplementation":
                continue
            with self.subTest(method=d.name):
                self.assertTrue(
                    (d / "train_graflag.py").is_file(),
                    f"{d.name} is declared a reimplementation but has no "
                    f"train_graflag.py to be the implementation")


class UpstreamSources(unittest.TestCase):
    """A method whose upstream repository has moved cannot build.

    Network-dependent, so skipped unless GRAFLAG_CHECK_UPSTREAM is set.
    """

    def test_clone_urls_resolve(self):
        import os
        if not os.environ.get("GRAFLAG_CHECK_UPSTREAM"):
            self.skipTest("set GRAFLAG_CHECK_UPSTREAM=1 to check the network")

        import urllib.request
        # Read the URL from the .env, not the Dockerfile: the Dockerfiles clone
        # `${SOURCE_CODE}` now, so a regex over them would find nothing and
        # this test would pass by checking zero URLs. The .env is also the only
        # place gady, slade and generaldyg ever named theirs, so three of the
        # seven were never reached here in the first place.
        urls = set()
        for d in method_dirs():
            if d.name == "example" or not clones(d):
                continue
            url = parse_env(d / ".env").get("SOURCE_CODE", "")
            if url.startswith("https://"):
                urls.add((d.name, url))

        self.assertEqual(len(urls), len(CLONING_METHODS), "a cloning method has no https SOURCE_CODE")
        for name, url in sorted(urls):
            with self.subTest(method=name, url=url):
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    self.assertLess(resp.status, 400)


if __name__ == "__main__":
    unittest.main()


class ResultWriterOutputDir(unittest.TestCase):
    """`ResultWriter(str(p.experiment))` writes results.json to the wrong place.

    ExperimentPaths.experiment is the experiment directory's *name*, not its
    path (that is `.exp`). Handing it to ResultWriter creates a relative
    directory of that name inside the container: the method exits 0, and the
    runner fails the run with "Method exited 0 but wrote no results.json".
    reference/sdk.md showed this form, so it reached two integrations before
    anything noticed.
    """

    def test_result_writer_is_not_given_the_experiment_name(self):
        for d in method_dirs():
            script = d / "train_graflag.py"
            if not script.is_file():
                continue
            tree = ast.parse(script.read_text(errors="replace"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == "ResultWriter"):
                    continue
                for arg in node.args:
                    attrs = {n.attr for n in ast.walk(arg)
                             if isinstance(n, ast.Attribute)}
                    with self.subTest(method=d.name):
                        self.assertNotIn(
                            "experiment", attrs,
                            "ResultWriter got ExperimentPaths.experiment (a "
                            "name); pass nothing (defaults to $EXP) or p.exp")
