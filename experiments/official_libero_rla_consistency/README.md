# Official LIBERO RLA consistency integration audit

This branch adds a minimal objective wrapper and fail-closed data audit, not a replacement policy trainer. See official_recipe_audit.md and minimal_pipeline_report.md for incomplete gates.

Use the immutable official checkout and preserve its license. This repository retains the original source attribution/license; downloaded weights/data are excluded.

Environment: RLA Python with torch2.8/transformers4.57 for CPU checkpoint audit; use the separately pinned official policy environment described in official_recipe_audit.md for future VLA tests. No environment upgrades performed.

```bash
export HF_HOME="$DATA_CACHE/hf" TORCH_HOME="$DATA_CACHE/torch"
export XDG_CACHE_HOME="$DATA_CACHE/xdg" TMPDIR="$DATA_CACHE/tmp"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
python check_checkpoint.py --official "$OFFICIAL_CHECKOUT" --run "$RELEASED_AE_RUN" --output "$RUN_DIR/checkpoint_acceptance.json"
RLA_PROJECT_ROOT="$PROJECT_ROOT" RUN_DIR="$RUN_DIR" python data_audit.py
```

Use new output directories; scripts refuse conflicting output. Paths are environment placeholders, not bundled data. Source data_audit assumes existing historic manifests with their stable relative project locations.

