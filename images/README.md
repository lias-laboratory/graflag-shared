# images/

Docker images shared by more than one method.

A method normally builds `methods/<name>/Dockerfile`. When several methods
would build the same image, they declare `IMAGE=<name>` in their `.env`
instead and the build uses `images/<name>/Dockerfile`. `graflag` resolves the
name once, in `DockerManager._image_names`, so the image that is built and the
image the service runs cannot disagree.

| Image | Used by | Why it is shared |
|---|---|---|
| `bond_base` | the 17 `bond_*` methods | Their Dockerfiles were byte-identical. The detector is chosen at run time from `METHOD_NAME`, not baked in, so one build serves all of them -- and 17 copies of a ~9 GB image did not fit on the cluster. |

Two things to keep true of anything added here:

- **No method-specific step.** If the image needs to know which method it is,
  it is not shared; that knowledge belongs in the `.env` and reaches the
  container as an environment variable.
- **The build context is `SHARED_DIR`**, exactly as for a method Dockerfile,
  so `COPY` paths are written `images/<name>/...` or `libs/...`.

This directory is part of the Docker build context (`.dockerignore` lists
`datasets/` and `experiments/`, not this), so it needs no separate handling --
but do not rename it to `build/`, which that file does list.

Those exclusions only apply when `.dockerignore` is on the share, and nothing
in GraFlag puts it there: `graflag sync` copies a method directory and
`sync --lib` copies a library, neither touches the root. `graflag copy -s
./.dockerignore --dest .` from this checkout is what makes the file take
effect; `build_method_image()` warns in `build.log` when it is missing.
