"""
XIII experiment-registry demo — a queryable ledger of falsified experiments.

The Thirteenth Man kills most hypotheses. This turns that graveyard into a
structured, queryable record: seed it, query it, and let it audit its own
integrity. All data below is synthetic.
"""
import xiii
from xiii import ExperimentRecord, ExperimentRegistry

# A handful of synthetic experiments (structured post-mortems).
SEED = [
    ExperimentRecord(
        id="EXP-001",
        title="Mean-reversion on EURUSD H1",
        status="KILLED_BY_13TH_MAN",
        date="2026-01-10",
        assets=["EURUSD"],
        theme="mean-reversion",
        metrics={"sharpe": 0.4, "tstat": 0.9},
        fail_reason="edge dies once spread is charged; t-stat is noise.",
        source_memory="exp001-eurusd-meanrev",
        origins=["assistant-A", "research-partner"],
    ),
    ExperimentRecord(
        id="EXP-002",
        title="Trend filter on an index proxy",
        status="DEPLOYED",
        date="2026-02-01",
        assets=["US100"],
        theme="trend",
        metrics={"sharpe": 0.6, "maxdd": -0.09},
        deployed=True,
        source_memory="exp002-index-trend",
        origins=["research-partner"],
    ),
    ExperimentRecord(
        id="EXP-003",
        title="Event-window macro edge",
        status="INCONCLUSIVE",
        date="2026-02-15",
        assets=["GBPUSD"],
        theme="macro",
        metrics={"tstat": 1.3, "n": 22},
        notes="sample too small to conclude; not refuted, not confirmed.",
        source_memory="exp003-macro-event",
        origins=["assistant-A"],
    ),
]


def main():
    reg = ExperimentRegistry(SEED)

    print("all experiments:", len(reg))
    print("KILLED:", [r.id for r in reg.query(status="KILLED_BY_13TH_MAN")])
    print("touching GBPUSD:", [r.id for r in reg.query(asset="GBPUSD")])
    print("live:", [r.id for r in reg.query(deployed=True)])
    print("suggested by research-partner:", [r.id for r in reg.query(origin="research-partner")])

    # Who suggested what, and how deployable it turned out.
    print()
    reg.print_origins()

    # The Thirteenth Man, turned on the ledger itself.
    print()
    ok = reg.print_verify()
    print("\nledger auditable:", ok)


if __name__ == "__main__":
    main()
