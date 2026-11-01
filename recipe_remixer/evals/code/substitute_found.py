"""Code evaluator for the get_substitute tool: did it find a swap?

A miss is the tool answering "No substitute found for X." Misses show which
ingredients real requests bring that the tool can't swap, or names the agent
asks about that the tool doesn't recognize.

The same file runs in two places:
- Online, LangSmith runs perform_eval on every get_substitute run in the
  tracing project. Online code evaluators can only use the standard library,
  so this file imports nothing. recipe_remixer/evals/online.py uploads it.
- Offline, substitute_found is an evaluate() evaluator. It finds the
  get_substitute runs inside each experiment run.
"""

MISS = "No substitute found"


def _found(output) -> bool:
    # A stored trace has the tool message as a dict. During evaluate() it's
    # still a ToolMessage object.
    content = output.get("content") if isinstance(output, dict) else getattr(output, "content", output)
    return not str(content).startswith(MISS)


# On a tracing project LangSmith calls this with the run only, since there's
# no dataset example.
def perform_eval(run, example=None):
    return {"substitute_found": int(_found(run["outputs"]["output"]))}


def _tool_runs(run):
    for child in run.child_runs or []:
        if child.name == "get_substitute":
            yield child
        yield from _tool_runs(child)


def substitute_found(run) -> dict:
    calls = list(_tool_runs(run))
    misses = [f"{c.inputs['ingredient']} ({c.inputs['diet']})" for c in calls if not _found(c.outputs["output"])]
    result = {"score": 1 - len(misses) / len(calls) if calls else 1.0}
    if misses:
        result["comment"] = "No substitute found for: " + ", ".join(misses)
    return result
