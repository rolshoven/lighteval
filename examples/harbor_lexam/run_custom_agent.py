import argparse
import json
from pathlib import Path


def run_agent(sample: dict) -> dict:
    """Template for external Harbor runner implementations.

    Replace this with your own agent stack (tool calling, skills, memory, etc.).
    """
    prompt = sample["prompt"]
    task_name = sample["task_name"]
    choices = sample.get("choices", [])

    # Placeholder answer for demonstration only.
    if task_name.startswith("lexam_mcq_"):
        answer_letter = choices[0] if choices else "A"
        text = f"Reasoning omitted.\nFinal Answer: ###{answer_letter}###"
    else:
        text = f"Template answer for task {task_name}.\n\nPrompt was:\n{prompt[:200]}"

    return {
        "text": text,
        "reasoning": None,
        "metrics": {"tool_calls": 0.0},
    }


def main():
    parser = argparse.ArgumentParser(description="External Harbor runner template for LEXam.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    sample = json.loads(Path(args.input_path).read_text())
    output = run_agent(sample)
    Path(args.output_path).write_text(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
