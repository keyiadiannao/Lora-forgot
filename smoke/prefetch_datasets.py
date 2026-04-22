import argparse
import json
import os
from typing import Any, Dict, List, Optional

from datasets import DownloadConfig, load_dataset


ITEMS = [
    {"dataset": "gsm8k", "subset": "main", "splits": ["train", "test"]},
    {"dataset": "winogrande", "subset": "winogrande_s", "splits": ["train", "validation"]},
    {"dataset": "imdb", "subset": None, "splits": ["train", "test"]},
    {"dataset": "hotpotqa/hotpot_qa", "subset": "distractor", "splits": ["train", "validation"]},
    {"dataset": "ohjoonhee/2WikiMultihopQA", "subset": None, "splits": ["train", "validation"]},
    {"dataset": "cais/mmlu", "subset": "high_school_physics", "splits": ["dev", "test"]},
]


def _apply_endpoint(endpoint: Optional[str]) -> None:
    if not endpoint:
        return
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_ENDPOINT"] = endpoint
    os.environ["HUGGINGFACE_HUB_BASE_URL"] = endpoint
    try:
        import datasets.config as ds_cfg  # type: ignore

        ds_cfg.HF_ENDPOINT = endpoint
    except Exception:
        pass
    try:
        from huggingface_hub import constants as hf_const  # type: ignore

        hf_const.ENDPOINT = endpoint
    except Exception:
        pass


def run_prefetch(local_files_only: bool, endpoint: Optional[str]) -> Dict[str, Any]:
    _apply_endpoint(endpoint)
    rows: List[Dict[str, Any]] = []
    for it in ITEMS:
        ds_name = it["dataset"]
        subset = it["subset"]
        ok = True
        msg = "ok"
        loaded: Dict[str, Optional[int]] = {}
        try:
            dl_cfg = DownloadConfig(local_files_only=local_files_only)
            if subset:
                ds = load_dataset(ds_name, subset, download_config=dl_cfg)
            else:
                ds = load_dataset(ds_name, download_config=dl_cfg)
            for sp in it["splits"]:
                loaded[sp] = len(ds[sp]) if sp in ds else None
        except Exception as e:
            ok = False
            msg = f"{type(e).__name__}: {e}"
        rows.append(
            {
                "dataset": ds_name,
                "subset": subset,
                "required_splits": it["splits"],
                "loaded_lengths": loaded,
                "ok": ok,
                "message": msg[:500],
            }
        )
    return {"endpoint": endpoint, "local_files_only": local_files_only, "results": rows}


def main() -> None:
    p = argparse.ArgumentParser(description="Prefetch required datasets for 8-pair smoke run")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--endpoint", type=str, default="")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()

    report = run_prefetch(local_files_only=bool(args.local_files_only), endpoint=args.endpoint or None)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] report: {args.output}")
    for r in report["results"]:
        state = "OK" if r["ok"] else "FAIL"
        print(f"{state} {r['dataset']} subset={r['subset']} loaded={r['loaded_lengths']} msg={r['message']}")


if __name__ == "__main__":
    main()
