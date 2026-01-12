import argparse
import json
import subprocess
import sys
import os
from pathlib import Path


def run_step(command, step_name, env_override=None):
    """
    Executes a shell command and streams the output.
    """
    print(f"\n{'=' * 60}\nSTEP: {step_name}\nCMD:  {command}\n{'=' * 60}")

    # Prepare environment variables
    env = os.environ.copy()
    if env_override:
        env.update(env_override)

    try:
        # executable='/bin/bash' is often required for specific shell features
        subprocess.run(command, shell=True, check=True, env=env, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR in step '{step_name}': Exit Code {e.returncode}")
        sys.exit(e.returncode)


def run_mli_pipeline(input_dir, output_dir, config):
    print("🚀 Starting MLI (Multi-Label) Pipeline")

    # Extract basic configuration for detection
    # We keep these because they are essential for HPC/GPU control
    batch_size = config.get("batch_size", 1)
    device = config.get("device", "cuda")

    # 1. Detection (GPU)
    # Input: Raw Images -> Output: Cropped Labels + CSV
    run_step(
        f"python scripts/processing/detection.py -j {input_dir} -o {output_dir} --batch-size {batch_size} --device {device}",
        "Segmentation/Detection"
    )

    # 2. Empty Labels (CPU)
    # Input: output_dir/input_cropped
    run_step(
        f"python scripts/processing/analysis.py -o {output_dir} -i {output_dir}/input_cropped",
        "Empty Labels Analysis",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )

    # 3. NURI Classifier (CPU Forced)
    # Input: output_dir/not_empty
    run_step(
        f"python scripts/processing/classifiers.py -m 1 -j {output_dir}/not_empty -o {output_dir}",
        "Classification: NURI",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )

    # 4. HP Classifier (CPU Forced)
    # Input: output_dir/not_identifier
    run_step(
        f"python scripts/processing/classifiers.py -m 2 -j {output_dir}/not_identifier -o {output_dir}",
        "Classification: Handwritten/Printed",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )

    # 5. Tesseract (CPU)
    # Input: output_dir/printed
    run_step(
        f"python scripts/processing/tesseract.py -d {output_dir}/printed -o {output_dir}",
        "Tesseract OCR",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )

    # 6. Postprocessing (CPU Forced)
    # Input: output_dir/ocr_preprocessed.json
    run_step(
        f"python scripts/postprocessing/process.py -j {output_dir}/ocr_preprocessed.json -o {output_dir}",
        "Postprocessing",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )


def run_sli_pipeline(input_dir, output_dir, config):
    print("🚀 Starting SLI (Single-Label) Pipeline")

    # 1. Empty Labels (CPU)
    # Input: Raw Input Images (already cropped)
    run_step(
        f"python scripts/processing/analysis.py -o {output_dir} -i {input_dir}",
        "Empty Labels Analysis",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )

    # 2. NURI Classifier (CPU Forced)
    run_step(
        f"python scripts/processing/classifiers.py -m 1 -j {output_dir}/not_empty -o {output_dir}",
        "Classification: NURI",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )

    # 3. HP Classifier (CPU Forced)
    run_step(
        f"python scripts/processing/classifiers.py -m 2 -j {output_dir}/not_identifier -o {output_dir}",
        "Classification: Handwritten/Printed",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )

    # 4. Rotation (SLI Specific)
    # Input: output_dir/printed -> Output: output_dir/rotated
    run_step(
        f"python scripts/processing/rotation.py -o {output_dir}/rotated -i {output_dir}/printed",
        "Image Rotation",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )

    # 5. Tesseract (CPU)
    # Input: output_dir/rotated
    run_step(
        f"python scripts/processing/tesseract.py -d {output_dir}/rotated -o {output_dir}",
        "Tesseract OCR",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )

    # 6. Postprocessing (CPU Forced)
    run_step(
        f"python scripts/postprocessing/process.py -j {output_dir}/ocr_preprocessed.json -o {output_dir}",
        "Postprocessing",
        env_override={"CUDA_VISIBLE_DEVICES": ""}
    )


def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline Entrypoint")
    parser.add_argument("--input-dir", required=True, help="Path to input images")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    parser.add_argument("--config-json", default="{}", help="JSON string with configuration")

    args = parser.parse_args()

    # Resolve absolute paths (crucial for internal script calls)
    input_dir = str(Path(args.input_dir).resolve())
    output_dir = str(Path(args.output_dir).resolve())

    # Parse JSON
    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON config: {e}")
        sys.exit(1)

    print(f"🔧 Config loaded: {json.dumps(config, indent=2)}")

    # Determine pipeline type (Default: 'mli')
    # Checks for 'pipeline_type' or 'type' in JSON
    pipeline_type = config.get("pipeline_type", config.get("type", "mli")).lower()

    if pipeline_type == "mli":
        run_mli_pipeline(input_dir, output_dir, config)
    elif pipeline_type == "sli":
        run_sli_pipeline(input_dir, output_dir, config)
    else:
        print(f"❌ Unknown pipeline type: {pipeline_type}. Supported: 'mli', 'sli'")
        sys.exit(1)

    print("\n✅ Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
