"""Optional Inspect task. Use a local model provider."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, scorer, accuracy
from inspect_ai.solver import TaskState, generate
from evaluation.challenge import benchmark
from evaluation.score import verify_quantity


@scorer(metrics=[accuracy()])
def si_quantity():
    async def score(state: TaskState, target: Target):
        value, unit = target.text.split()
        verdict = verify_quantity(state.output.completion, value, unit)
        return Score(value=int(verdict == "pass"), explanation=verdict)

    return score


@task
def engineering_quantities():
    cases = [Sample(id=r["id"], input=r["prompt"], target=f"{r['value']} {r['unit']}") for r in benchmark()]
    return Task(dataset=cases, solver=generate(), scorer=si_quantity())
