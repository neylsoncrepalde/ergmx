# Quick start

Two complete analyses, each from the data to a table of results, to read in
a few minutes and adapt to your own networks:

- **An ERGM**: the friendships of a high school. Do students befriend others
  like them, and the friends of their friends?
- **A multilevel ERGM**: researchers, their laboratories, and the ties
  within and between the two levels, simulated from a known model and drawn
  with [multinets](https://github.com/neylsoncrepalde/multinets-py).

Each step links to the page of the user guide that covers it in depth. To
run them, add `ergmx` with igraph and matplotlib to a uv project (created
with `uv init`), and multinets for the second
(see [Installation](../installation.md)):

```bash
uv add "ergmx[igraph,plot]"
uv add "multinets @ git+https://github.com/neylsoncrepalde/multinets-py"
```

Then run a script with `uv run python analysis.py`.

```{toctree}
:maxdepth: 1

ergm
multilevel
```
