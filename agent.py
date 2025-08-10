from __future__ import annotations

import abc
import asyncio
import time
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Protocol, Iterable

# -----------------------------
# Core types
# -----------------------------


@dataclass
class ModelSpec:
    name: str
    provider: str  # e.g., "openai:gpt-4o", "anthropic:sonnet"
    cost_per_1k_input: float
    cost_per_1k_output: float
    rpm_limit: Optional[int] = None  # requests per minute
    tpm_limit: Optional[int] = None  # tokens per minute
    max_output_tokens: int = 512
    tier: int = 0  # 0=cheap, 1=mid, 2=premium
    skills: Tuple[str, ...] = ()  # declared strengths ("code","math","summarize")


@dataclass
class CallRequest:
    system: str
    user: str
    temperature: float = 0.2
    stop: Optional[Sequence[str]] = None
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # plan/skill/constraints


@dataclass
class CallResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float
    raw: Any = None  # provider payload if needed


class LLMAdapter(Protocol):
    spec: ModelSpec

    async def acomplete(self, req: CallRequest) -> CallResult: ...


# -----------------------------
# Planning
# -----------------------------


@dataclass
class PlanStep:
    skill: str  # e.g., "reason", "summarize", "code"
    k_models: int = 2  # number of debaters for this step
    tier_hint: Optional[int] = None  # min tier to consider
    max_rounds: int = 1  # improve rounds allowed by verifier


@dataclass
class Plan:
    steps: List[PlanStep]
    hard_budget_usd: float
    hard_latency_s: Optional[float] = None
    seed: int = 123


class Planner(abc.ABC):
    @abc.abstractmethod
    async def make_plan(
        self, task: str, budget_usd: float, latency_s: Optional[float]
    ) -> Plan: ...


class HeuristicPlanner(Planner):
    def __init__(self, k: int = 2):
        self.k = k

    async def make_plan(
        self, task: str, budget_usd: float, latency_s: Optional[float]
    ) -> Plan:
        # Very light heuristic: detect code/math by tokens; else generic reason+summarize
        contains_digits = any(ch.isdigit() for ch in task)
        skill = "math" if contains_digits else "reason"
        steps = [PlanStep(skill=skill, k_models=2, tier_hint=0, max_rounds=1)]
        return Plan(steps=steps, hard_budget_usd=budget_usd, hard_latency_s=latency_s)


# -----------------------------
# Router
# -----------------------------


@dataclass
class RouterCfg:
    epsilon: float = 0.05  # exploration (for static policy)
    alpha_cost: float = 0.0  # tradeoff: reward − alpha·cost (if used)


@dataclass
class ArmStat:
    pulls: int = 0
    winrate: float = 0.0
    cost: float = 0.0

    def update(self, reward: float, cost: float) -> None:
        self.pulls += 1
        self.winrate += (reward - self.winrate) / self.pulls
        self.cost += (cost - self.cost) / self.pulls


class Router:
    def __init__(
        self,
        adapters: Dict[str, LLMAdapter],
        cfg: RouterCfg = RouterCfg(),
        seed: int = 123,
    ):
        self.adapters = adapters
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.stats: Dict[str, ArmStat] = {name: ArmStat() for name in adapters}

    def _score(self, name: str) -> float:
        s = self.stats[name]
        explore = self.rng.random() < self.cfg.epsilon
        if explore or s.pulls == 0:
            return self.rng.random()
        # Default static utility: winrate − alpha·cost
        return s.winrate - self.cfg.alpha_cost * s.cost

    def pick_k(self, skill: str, k: int, tier_hint: Optional[int] = None) -> List[str]:
        cand = [
            a
            for a in self.adapters.values()
            if (tier_hint is None or a.spec.tier >= tier_hint)
            and (not skill or skill in a.spec.skills or not a.spec.skills)
        ]
        if not cand:
            cand = list(self.adapters.values())
        scored = sorted(cand, key=lambda a: self._score(a.spec.name), reverse=True)
        return [a.spec.name for a in scored[: max(1, k)]]

    def update(self, name: str, reward: float, cost: float) -> None:
        self.stats[name].update(reward, cost)


# -----------------------------
# Debate, Judge, Verify
# -----------------------------


@dataclass
class Candidate:
    model: str
    text: str
    latency_s: float
    cost_usd: float
    tokens_in: int
    tokens_out: int


class Debate:
    def __init__(self, adapters: Dict[str, LLMAdapter]):
        self.adapters = adapters

    async def run(self, names: List[str], req: CallRequest) -> List[Candidate]:
        async def one(name: str) -> Candidate:
            res = await self.adapters[name].acomplete(req)
            return Candidate(
                name,
                res.text,
                res.latency_s,
                res.cost_usd,
                res.tokens_in,
                res.tokens_out,
            )

        return await asyncio.gather(*[one(n) for n in names])


class Judge(abc.ABC):
    @abc.abstractmethod
    async def pick(
        self, task: str, candidates: List[Candidate]
    ) -> Tuple[int, Dict[str, Any]]: ...


class LLMJudge(Judge):
    def __init__(self, judge_adapter: LLMAdapter):
        self.judge = judge_adapter

    async def pick(
        self, task: str, candidates: List[Candidate]
    ) -> Tuple[int, Dict[str, Any]]:
        # Simple ranking prompt; candidates are short. Implement your own format.
        listing = "".join([f"[#{i}]{c.text}" for i, c in enumerate(candidates)])
        req = CallRequest(
            system="You are a strict evaluator.",
            user=f"Task: {task} Choose the best numbered answer and provide a short rationale.{listing}",
        )
        res = await self.judge.acomplete(req)
        # Extract first index occurrence like "#1" or "[1]" (engineers can tighten this)
        chosen = 0
        for i in range(len(candidates)):
            if f"#{i}" in res.text or f"[{i}]" in res.text:
                chosen = i
                break
        return chosen, {"judge_text": res.text}


class Verifier(abc.ABC):
    @abc.abstractmethod
    async def check(
        self, task: str, answer: str, metadata: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]: ...


class LLMVerifier(Verifier):
    def __init__(self, verifier_adapter: LLMAdapter):
        self.ver = verifier_adapter

    async def check(
        self, task: str, answer: str, metadata: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        req = CallRequest(
            system="You are a strict verifier.",
            user=f"Task: {task} Answer: {answer} Verify factual correctness, internal consistency, and instruction following. Reply ACCEPT or REVISE with a short reason.",
        )
        res = await self.ver.acomplete(req)
        ok = "ACCEPT" in res.text.upper()
        return ok, {"verifier_text": res.text}


# -----------------------------
# Budget manager & traces
# -----------------------------


@dataclass
class Budget:
    usd_left: float
    deadline_s: Optional[float] = None

    def allow(self, cost: float, latency: float) -> bool:
        time_ok = True if self.deadline_s is None else (latency <= self.deadline_s)
        money_ok = cost <= self.usd_left
        return time_ok and money_ok

    def charge(self, cost: float) -> None:
        self.usd_left = max(0.0, self.usd_left - cost)


@dataclass
class StepTrace:
    step_idx: int
    skill: str
    chosen_models: List[str]
    candidates: List[Candidate]
    chosen_idx: int
    judge_meta: Dict[str, Any]
    verified: bool
    verifier_meta: Dict[str, Any]


@dataclass
class RunTrace:
    task: str
    final_text: str
    steps: List[StepTrace]
    total_cost_usd: float
    total_latency_s: float


# -----------------------------
# Orchestrator
# -----------------------------


class Agent:
    def __init__(
        self,
        adapters: Dict[str, LLMAdapter],
        planner: Planner,
        router: Router,
        judge: Judge,
        verifier: Verifier,
        seed: int = 123,
    ):
        self.adapters = adapters
        self.planner = planner
        self.router = router
        self.judge = judge
        self.verifier = verifier
        self.rng = random.Random(seed)

    async def run(
        self, task: str, budget_usd: float, latency_s: Optional[float] = None
    ) -> RunTrace:
        plan = await self.planner.make_plan(task, budget_usd, latency_s)
        budget = Budget(usd_left=plan.hard_budget_usd, deadline_s=plan.hard_latency_s)
        total_cost = 0.0
        total_lat = 0.0
        step_traces: List[StepTrace] = []
        user_req = CallRequest(system="You are a helpful assistant.", user=task)

        for si, step in enumerate(plan.steps):
            names = self.router.pick_k(
                skill=step.skill, k=step.k_models, tier_hint=step.tier_hint
            )
            # Debate (parallel)
            deb = Debate(self.adapters)
            cand = await deb.run(names, user_req)
            step_cost = sum(c.cost_usd for c in cand)
            step_lat = max((c.latency_s for c in cand), default=0.0)
            if not budget.allow(step_cost, step_lat):
                # If over budget, trim to cheapest single model
                cheapest = min(cand, key=lambda c: c.cost_usd)
                cand = [cheapest]
                step_cost = cheapest.cost_usd
                step_lat = cheapest.latency_s
            total_cost += step_cost
            total_lat += step_lat
            budget.charge(step_cost)

            # Judge
            j_idx, j_meta = await self.judge.pick(task, cand)
            chosen = cand[j_idx]

            # Verifier (optionally one improve round)
            ok, v_meta = await self.verifier.check(
                task, chosen.text, {"skill": step.skill}
            )
            verified = ok
            if (not ok) and step.max_rounds > 0 and budget.usd_left > 0.0:
                # Single improve round: escalate tier by +1 if available
                next_tier = max(self.adapters[n].spec.tier for n in names) + 1
                names2 = self.router.pick_k(skill=step.skill, k=1, tier_hint=next_tier)
                cand2 = await Debate(self.adapters).run(
                    names2,
                    CallRequest(
                        system="Improve the answer.",
                        user=f"Task: {task} Previous answer: {chosen.text} Fix issues succinctly.",
                    ),
                )
                total_cost += sum(c.cost_usd for c in cand2)
                total_lat += max((c.latency_s for c in cand2), default=0.0)
                budget.charge(sum(c.cost_usd for c in cand2))
                # Judge between old and improved
                all_cand = [chosen] + cand2
                j_idx2, j_meta2 = await self.judge.pick(task, all_cand)
                chosen = all_cand[j_idx2]
                ok2, v_meta2 = await self.verifier.check(
                    task, chosen.text, {"skill": step.skill, "round": 2}
                )
                verified = ok2
                # merge judge/verifier meta
                j_meta = {**j_meta, "improve": j_meta2}
                v_meta = {**v_meta, "improve": v_meta2}

            # Update router feedback with a very cheap binary reward (engineers can replace)
            self.router.update(
                chosen.model, reward=1.0 if verified else 0.0, cost=chosen.cost_usd
            )

            step_traces.append(
                StepTrace(
                    step_idx=si,
                    skill=step.skill,
                    chosen_models=names,
                    candidates=cand,
                    chosen_idx=j_idx,
                    judge_meta=j_meta,
                    verified=verified,
                    verifier_meta=v_meta,
                )
            )

        final_text = (
            step_traces[-1].candidates[step_traces[-1].chosen_idx].text
            if step_traces
            else ""
        )
        return RunTrace(
            task=task,
            final_text=final_text,
            steps=step_traces,
            total_cost_usd=total_cost,
            total_latency_s=total_lat,
        )
