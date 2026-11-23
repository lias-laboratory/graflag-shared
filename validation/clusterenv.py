"""The private half of the validation scripts: where the cluster is.

Where these experiments ran is not published. Results name the machines
node-1 (the manager), node-2 and node-3, and every real value -- addresses,
hostnames -- lives in validation/cluster.env, which is untracked (template:
cluster.env.example).

The scripts that write tracked results read node names straight from the
cluster (`docker node ls`, the runner's own record), so they go through
`alias()` here, and `assert_private()` refuses to write a file that still
contains any real value from cluster.env. Writing first and scrubbing after is
how real hostnames reached the results once already.
"""

from pathlib import Path

PATH = Path(__file__).resolve().parent / "cluster.env"


def cluster_env(path=PATH):
    """validation/cluster.env as a dict. Missing is an error, not a default."""
    if not path.is_file():
        raise SystemExit(f"[ERROR] {path} is missing; copy cluster.env.example "
                         "and fill it in")
    values = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"')
    return values


def aliases(env=None):
    """{real hostname: node-N} from NODE_ALIASES="host=node-1 host=node-2 ..."."""
    env = env if env is not None else cluster_env()
    raw = env.get("NODE_ALIASES", "")
    pairs = dict(p.split("=", 1) for p in raw.split() if "=" in p)
    if not pairs:
        raise SystemExit("[ERROR] cluster.env has no NODE_ALIASES; results would "
                         "carry the nodes' real hostnames")
    return pairs


def alias(node, table):
    """The published name of `node`. An unknown real name is an error."""
    if node is None or node in table.values():
        return node
    if node not in table:
        raise SystemExit(f"[ERROR] node {node!r} has no entry in cluster.env "
                         "NODE_ALIASES; refusing to publish it")
    return table[node]


def private_values(env=None):
    """Every real value in cluster.env that must not appear in a result."""
    env = env if env is not None else cluster_env()
    values = set()
    for key in ("MANAGER", "NODES", "WORKERS", "REGISTRY"):
        values.update(v.split(":")[0] for v in env.get(key, "").split())
    values.update(aliases(env))
    return {v for v in values if v}


def assert_private(text, where, env=None):
    """Raise if `text`, about to be written to `where`, names the cluster."""
    found = sorted(v for v in private_values(env) if v in text)
    if found:
        raise SystemExit(f"[ERROR] {where} would contain {len(found)} private "
                         "value(s) from cluster.env; not written")
