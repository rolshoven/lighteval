"""Subprocess runner for `lighteval harbor run` (LiteLLM baseline, no Harbor sandbox)."""

import argparse
import json
import os

from lighteval.tasks.multilingual.tasks.swiss_legal.lexam_harbor_common import (
    build_sample_output,
    run_litellm_baseline,
)


def main():
    parser = argparse.ArgumentParser(description="LEXam subprocess runner for lighteval harbor run.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    model_name = (
        os.getenv("HARBOR_REFERENCE_MODEL")
        or os.getenv("REFERENCE_MODEL")
        or os.getenv("LITELLM_MODEL")
        or "openai/gpt-4o-mini"
    )

    with open(args.input_path, "r") as f:
        sample = json.load(f)

    text, metrics = run_litellm_baseline(sample=sample, model_name=model_name)
    output = build_sample_output(sample=sample, text=text, metrics=metrics)

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
