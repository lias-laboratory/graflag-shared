#!/usr/bin/env python3
"""Check that a finished experiment published a result worth believing.

Gate 4 of the method-integration skill. The checks moved into the GraFlag
client in 1.2.0 (`graflag/verify.py` in the graflag repository), where
`graflag verify -e EXP` and the MCP server's `verify_experiment` tool reach
them. This script keeps the command line it always had:

    python3 verify_run.py exp__method__dataset__timestamp
    python3 verify_run.py --config path/to/config.env exp__...
    python3 verify_run.py --json exp__...

Exit status is 1 if any check failed, 0 otherwise (warnings do not fail).
"""

import sys

try:
    from graflag.verify import main
except ImportError:                                        # graflag < 1.2.0
    sys.exit("[ERROR] this check needs graflag 1.2.0 or later: "
             "`pip install -U graflag`, or `cd graflag && pip install -e .`")

if __name__ == "__main__":
    sys.exit(main())
