from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def _adapter_sha(snapshot: Path) -> str | None:
    path = snapshot / "adapter_model.safetensors"
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(*, source_root: Path, repo_id: str, output_dir: Path, config_path: Path, force: bool) -> None:
    source_root = source_root.resolve()
    if not (source_root / "nimble").is_dir():
        raise RuntimeError(f"Nimble source checkout is missing: {source_root}")
    sys.path.insert(0, str(source_root))

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=repo_id)).resolve()
    revision = snapshot.name
    contract_file = snapshot / "schema_config.json"
    contract = json.loads(contract_file.read_text(encoding="utf-8")) if contract_file.is_file() else {}

    if contract:
        from transformers import AutoTokenizer
        from nimble.training.candidate_schema import validate_contract

        validate_contract(contract, AutoTokenizer.from_pretrained(snapshot))

    if config_path.is_file() and output_dir.is_dir() and not force:
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved.get("model_id") == repo_id and saved.get("revision") == revision:
            print(f"Nimble prepared checkpoint already current: {output_dir}")
            return

    model_path = snapshot
    adapter_config = snapshot / "adapter_config.json"
    adapter_sha256 = _adapter_sha(snapshot)

    if adapter_config.is_file():
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        base_model = contract.get("model")
        base_revision = contract.get("revision")
        if not isinstance(base_model, str) or not base_model or not isinstance(base_revision, str) or not base_revision:
            raise RuntimeError("Nimble adapter is missing its pinned base model/revision contract")
        print(f"Preparing Nimble: merging {repo_id} into {base_model} on CPU")
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model,
            revision=base_revision,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        adapter = PeftModel.from_pretrained(base, snapshot)
        merged = adapter.merge_and_unload(safe_merge=True)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(output_dir)
        AutoTokenizer.from_pretrained(snapshot).save_pretrained(output_dir)
        if contract_file.is_file():
            shutil.copy2(contract_file, output_dir / "schema_config.json")
        ready = {
            "model": repo_id,
            "revision": revision,
            "base_revision": base_revision,
            "adapter_sha256": adapter_sha256,
        }
        (output_dir / "READY.json").write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
        model_path = output_dir

        del merged, adapter, base
    elif output_dir.resolve() != snapshot.resolve():
        # Current public release is an adapter. This branch keeps the installer usable
        # if a future release becomes a full checkpoint without needlessly copying it.
        model_path = snapshot

    config = {
        "model_path": str(model_path.resolve()),
        "model_id": repo_id,
        "revision": revision,
        "max_input_tokens": int(contract.get("max_length", 2048)),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Nimble ready: {model_path}")
    print(f"Nimble runtime config: {config_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a Bespoke Nimble checkpoint for Deqio.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(
        source_root=args.source_root,
        repo_id=args.repo_id,
        output_dir=args.output_dir.resolve(),
        config_path=args.config.resolve(),
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
