"""Entry point: pick a dataset, a clustering method, and an evaluation method, then run it."""

import argparse
import glob
import os

import kmean_default
import silhoutescore
import gapstatistics
import elbowmethod_kmean1
import davis_boulding_index

#only kmean_default.py is wired up as a clustering method for now, but new methods
#can be added here without touching the menu/CLI logic below
METHODS = {
    "kmeans_default": "From-scratch K-Means (kmean_default.py)",
}

DEFAULT_FRAMES_DIR = "kmeans_frames"
DEFAULT_VIDEO_PATH = "kmeans_training.gif"

DEFAULT_SILHOUETTE_PATH = "silhouette.png"
DEFAULT_GAP_PATH = "gap.png"
DEFAULT_ELBOW_PREFIX = "elbow"
DEFAULT_DAVIES_BOULDIN_PATH = "davies_bouldin.png"

#save_kwarg is the run() keyword used to save a plot: "save_path" for a single image,
#"save_prefix" for elbow's three images (named <prefix>_distortion/_inertia/_3d.<ext>)
EVALUATIONS = {
    "silhouette": ("Silhouette score (silhoutescore.py)", silhoutescore.run, "save_path"),
    "gap": ("Gap statistic (gapstatistics.py)", gapstatistics.run, "save_path"),
    "elbow": ("Elbow method - distortion/inertia (elbowmethod_kmean1.py)", elbowmethod_kmean1.run, "save_prefix"),
    "davies_bouldin": ("Davies-Bouldin index (davis_boulding_index.py)", davis_boulding_index.run, "save_path"),
}


#maps evaluation key -> the argparse destination that lets its save path be set non-interactively
SAVE_ARG_NAMES = {
    "silhouette": "save_silhouette",
    "gap": "save_gap",
    "elbow": "save_elbow_prefix",
    "davies_bouldin": "save_davies_bouldin",
}


def discover_datasets():
    """List JSON dataset files present in the working directory."""
    return sorted(f for f in glob.glob("*.json") if os.path.isfile(f))


def choose_dataset(preselected=None):
    if preselected:
        return preselected

    datasets = discover_datasets()
    if not datasets:
        return input("No .json datasets found in this folder. Enter a dataset path: ").strip()

    print("Available datasets:")
    for i, path in enumerate(datasets, start=1):
        print(f"  {i}. {path}")
    custom_idx = len(datasets) + 1
    print(f"  {custom_idx}. Enter a custom path")

    while True:
        choice = input(f"Choose dataset [1-{custom_idx}]: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(datasets):
                return datasets[idx - 1]
            if idx == custom_idx:
                return input("Dataset path: ").strip()
        print("Invalid choice, try again.")


def choose_method(preselected=None):
    if preselected:
        return preselected

    keys = list(METHODS.keys())
    print("Available clustering methods:")
    for i, key in enumerate(keys, start=1):
        print(f"  {i}. {METHODS[key]}")

    while True:
        choice = input(f"Choose method [1-{len(keys)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            return keys[int(choice) - 1]
        print("Invalid choice, try again.")


def choose_evaluations(preselected=None):
    if preselected:
        return list(EVALUATIONS.keys()) if preselected == "all" else [preselected]

    keys = list(EVALUATIONS.keys())
    print("Available evaluation methods:")
    for i, key in enumerate(keys, start=1):
        print(f"  {i}. {EVALUATIONS[key][0]}")
    all_idx = len(keys) + 1
    print(f"  {all_idx}. All of the above")

    while True:
        raw = input(f"Choose evaluation(s) [1-{all_idx}, comma-separated]: ").strip()
        picks = [p.strip() for p in raw.split(",") if p.strip()]
        if picks and all(p.isdigit() and 1 <= int(p) <= all_idx for p in picks):
            indices = [int(p) for p in picks]
            if all_idx in indices:
                return keys
            #de-duplicate while preserving the order the user typed them in
            seen = []
            for i in indices:
                key = keys[i - 1]
                if key not in seen:
                    seen.append(key)
            return seen
        print("Invalid choice, try again.")


def choose_training_output(args):
    """Work out the frames directory and video filename for kmean_default's training
    run. CLI flags always win; otherwise, if the method itself was picked interactively,
    ask the user for both (falling back to the defaults on a blank answer)."""
    frames_dir = args.frames_dir
    video_path = args.video
    if args.method is None:
        if frames_dir is None:
            frames_dir = input(
                f"Frames directory for kmean_default's training run [{DEFAULT_FRAMES_DIR}]: "
            ).strip() or DEFAULT_FRAMES_DIR
        if video_path is None:
            video_path = input(
                f"Video filename for kmean_default's training run [{DEFAULT_VIDEO_PATH}]: "
            ).strip() or DEFAULT_VIDEO_PATH
    else:
        frames_dir = frames_dir or DEFAULT_FRAMES_DIR
        video_path = video_path or DEFAULT_VIDEO_PATH
    if not os.path.splitext(video_path)[1]:
        video_path += os.path.splitext(DEFAULT_VIDEO_PATH)[1]
    return frames_dir, video_path


def choose_save_paths(evaluations, args):
    """For each evaluation, work out the filename its plot(s) should be saved to (or
    None to skip saving). CLI flags always win; otherwise, if the evaluation list itself
    was picked interactively, ask the user one-by-one."""
    interactive = args.evaluation is None
    save_paths = {}
    for name in evaluations:
        label, _run_fn, save_kwarg = EVALUATIONS[name]
        cli_value = getattr(args, SAVE_ARG_NAMES[name])
        if cli_value:
            save_paths[name] = cli_value
        elif interactive:
            if save_kwarg == "save_prefix":
                prompt = (f"Save plots for '{label}'? Enter a filename prefix "
                           "(3 images will be saved as <prefix>_distortion/_inertia/_3d.png), "
                           "or press Enter to skip: ")
            else:
                prompt = f"Save plot for '{label}'? Enter a filename, or press Enter to skip: "
            save_paths[name] = input(prompt).strip() or None
        else:
            save_paths[name] = None
    return save_paths


def main():
    parser = argparse.ArgumentParser(description="Run k-means clustering and evaluate the result.")
    parser.add_argument("--dataset", help="Path to the dataset JSON file (skips the interactive prompt)")
    parser.add_argument("--method", choices=list(METHODS.keys()), help="Clustering method to use")
    parser.add_argument("--frames-dir", help=f"Directory for kmean_default's per-iteration frames (default: {DEFAULT_FRAMES_DIR})")
    parser.add_argument("--video", help=f"Filename for kmean_default's training video (default: {DEFAULT_VIDEO_PATH})")
    parser.add_argument("--evaluation", choices=list(EVALUATIONS.keys()) + ["all"],
                         help="Evaluation method to run (skips the interactive prompt)")
    parser.add_argument("--save-silhouette", dest="save_silhouette", default=DEFAULT_SILHOUETTE_PATH,
                         help=f"Filename to save the silhouette score plot to (default: {DEFAULT_SILHOUETTE_PATH})")
    parser.add_argument("--save-gap", dest="save_gap", default=DEFAULT_GAP_PATH,
                         help=f"Filename to save the gap statistic plot to (default: {DEFAULT_GAP_PATH})")
    parser.add_argument("--save-elbow-prefix", dest="save_elbow_prefix", default=DEFAULT_ELBOW_PREFIX,
                         help=f"Filename prefix for the 3 elbow-method plots (_distortion/_inertia/_3d) (default: {DEFAULT_ELBOW_PREFIX})")
    parser.add_argument("--save-davies-bouldin", dest="save_davies_bouldin", default=DEFAULT_DAVIES_BOULDIN_PATH,
                         help=f"Filename to save the Davies-Bouldin plot to (default: {DEFAULT_DAVIES_BOULDIN_PATH})")
    args = parser.parse_args()

    dataset_path = choose_dataset(args.dataset)
    method = choose_method(args.method)
    frames_dir, video_path = choose_training_output(args)
    evaluations = choose_evaluations(args.evaluation)
    save_paths = choose_save_paths(evaluations, args)

    print(f"\nDataset:     {dataset_path}")
    print(f"Method:      {METHODS[method]}")
    print(f"Frames dir:  {frames_dir}")
    print(f"Video:       {video_path}")
    print(f"Evaluations: {', '.join(EVALUATIONS[e][0] for e in evaluations)}\n")

    print(f"=== Training: {METHODS[method]} ===")
    kmean_default.run(dataset_path, frames_dir=frames_dir, video_path=video_path)
    print()

    results = {}
    for name in evaluations:
        label, run_fn, save_kwarg = EVALUATIONS[name]
        print(f"=== Running evaluation: {label} ===")
        kwargs = {save_kwarg: save_paths[name]} if save_paths[name] else {}
        results[name] = run_fn(dataset_path, **kwargs)
        print()

    return results


if __name__ == "__main__":
    main()
