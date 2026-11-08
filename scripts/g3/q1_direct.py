"""Direct LLM coding of Q1 (strong classical baseline) for random-stratum full-text papers (deviations.md).

Sentences are kept only when a classical-solver term co-occurs with a comparison or reference cue, which
removes most introductory literature mentions. The LLM reads them and records Q1 in Q1_DIRECT:
y = accuracy and/or cost compared against a conventional solver run by the authors;
p = conventional solver only provides the reference solution; n = neither.

    python scripts/g3/q1_direct.py show
"""
import re
import sys

import pandas as pd

from pinnreview.core import G3
from pinnreview.evidence import PATTERNS, body_of
from pinnreview.fulltext import TEXT_DIR, short

CUE = re.compile(r"compar|versus|\bvs\.?|faster|speed[\s-]?up|outperform|reference\s+solution|ground[\s-]?truth|"
                 r"exact\s+solution|obtained\s+(by|with|using)|computed\s+(by|with|using)|generated\s+(by|with|using)|"
                 r"baseline|benchmark", re.I)
SOLVER = re.compile(PATTERNS["classical_solver"] + r"|numerical\s+solver|traditional\s+(numerical\s+)?(method|solver)s?|"
                    r"pseudo[\s-]?spectral|direct\s+numerical\s+simulation|\bDNS\b|\bCFD\b", re.I)

# short id -> "y" / "p" / "n" (LLM reading of the targeted sentences; abstract used when it states a solver comparison)
Q1_DIRECT = {
    "W2774549550": "n", "W2908429425": "p", "W2971386244": "y", "W2903660960": "p", "W3091651555": "y",
    "W3021668893": "n", "W3034660118": "n", "W3008992944": "y", "W3020952481": "n", "W3129796829": "n",
    "W4297168433": "n", "W3162074777": "n", "W3172099976": "n", "W3124424416": "n", "W3176958767": "y",
    "W4307312380": "p", "W4304615543": "n", "W4300980636": "n", "W4210423197": "n", "W3175293984": "p",
    "W4224089288": "p", "W4320086389": "p", "W4224758974": "n", "W4310872819": "p", "W4379379581": "n",
    "W3110941460": "n", "W4309395231": "n", "W4304699316": "y", "W4319869606": "n", "W4387355756": "n",
    "W4386779736": "n", "W4385473635": "n", "W4385210391": "n", "W4361865663": "n", "W4388110391": "n",
    "W4386557166": "n", "W4388184815": "n", "W4386719917": "p", "W4386452655": "n", "W4321648921": "n",
    "W4327485924": "n", "W4385627137": "n", "W4387209459": "n", "W4368373811": "n", "W4388198208": "n",
    "W4389072232": "n", "W4383176111": "n", "W4404342818": "n", "W4399911111": "p", "W4410773630": "n",
    "W4393165920": "n", "W4390751062": "n", "W4391019874": "y", "W4401262836": "y", "W4392972009": "n",
    "W4399303623": "n", "W4404328761": "n", "W4403346014": "p", "W4402703748": "n", "W4400601303": "n",
    "W4391305778": "p", "W4400413219": "n", "W4402906248": "n", "W4406534214": "n", "W4405495729": "p",
    "W4404705446": "y", "W7109457710": "y", "W4416103169": "n", "W4409241656": "y", "W4407388189": "n",
    "W4408127921": "n", "W4414581552": "n", "W4415654544": "n", "W4416013073": "p", "W4415064807": "y",
    "W4414298978": "n", "W4410345526": "n", "W4407894935": "y", "W4406331680": "p", "W4415315936": "y",
    "W4415314976": "n", "W4413467063": "p", "W4413251953": "p", "W4417026690": "n", "W4414976420": "n",
    "W4417299851": "n", "W4416258139": "n", "W7215357072": "y", "W4417028114": "n", "W7114927338": "n",
    "W4413572747": "n", "W4409625022": "p", "W7123335795": "y", "W4409638788": "n", "W4414816969": "y",
    "W4414769507": "n", "W4410006112": "n", "W7220532053": "n", "W4417252783": "y", "W4416154508": "p",
    "W4414595577": "y", "W4414689385": "p", "W4416203815": "n", "W4414942905": "n", "W4392817028": "n",
}


def papers():
    core = pd.read_csv(G3 / "core_v3.csv")
    ev = pd.read_csv(G3 / "evidence_v3.csv")
    rnd = set(core.loc[core["stratum"] == "random", "id"])
    return ev.loc[ev["full_text"] & ev["id"].isin(rnd), "id"].tolist()


def sentences(pid, k=3, width=240):
    body = body_of((TEXT_DIR / f"{short(pid)}.txt").read_text(encoding="utf-8"))
    body = re.sub(r"\s+", " ", body)
    hits = [s for s in re.split(r"(?<=[.!?])\s+", body) if SOLVER.search(s) and CUE.search(s)]
    return [h[:width] for h in hits[:k]]


def show():
    for pid in papers():
        sid = short(pid)
        if sid in Q1_DIRECT:
            continue
        hits = sentences(pid)
        print(f"{sid}: " + (" || ".join(hits) if hits else "<no solver+cue sentence>"))


if __name__ == "__main__":
    show()
