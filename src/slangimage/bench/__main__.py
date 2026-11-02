# slangimage.bench CLI
#
#   slangimage-bench run --backend auto --out results_metal.json
#   slangimage-bench report benchmarks/results_*.json -o docs/BENCHMARKS.md
#
# (also runnable as: python -m slangimage.bench ...)

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="slangimage-bench", description="SlangImage standard benchmark suite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="run the suite on one backend, write JSON results")
    runp.add_argument("--backend", default="auto", help="auto|metal|cuda|vulkan")
    runp.add_argument("--out", required=True, help="output JSON path")
    runp.add_argument("--iters", type=int, default=None, help="timed iterations per case")
    runp.add_argument("--sizes", default=None, help="comma-separated sizes, e.g. 1024,2048")

    repp = sub.add_parser("report", help="merge result JSONs into a markdown report")
    repp.add_argument("results", nargs="+", help="results_*.json files")
    repp.add_argument("-o", "--out", default=None, help="output .md path (default: stdout)")
    repp.add_argument("--allow-incompatible", action="store_true",
                      help=("merge duplicate backends or results with different protocols, "
                            "versions, iteration counts, or suite cells (normally refused)"))

    args = ap.parse_args()

    if args.cmd == "run":
        from slangimage import bench

        sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None
        result = bench.run(
            backend=args.backend,
            iters=args.iters or bench.DEFAULT_ITERS,
            sizes=sizes,
        )
        bench.save(result, args.out)
    else:
        from slangimage.bench.report import render

        md = render(args.results, strict=not args.allow_incompatible)
        if args.out:
            with open(args.out, "w") as f:
                f.write(md)
            print(f"wrote {args.out}")
        else:
            print(md)


if __name__ == "__main__":
    main()
