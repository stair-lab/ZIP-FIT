#!/usr/bin/env python3

import os
import shutil
import subprocess

from huggingface_hub import login, whoami, create_repo, snapshot_download, Repository

def main():
    # ============= 0) Setup constants/paths (NO placeholders) =============
    HF_TOKEN_FILE = os.path.expanduser("~/keys/master_hf_token.txt")
    REPO_ID = "UDACA/gemma-2-2b"  # new private repo
    LOCAL_REPO_DIR = os.path.expanduser("~/tmp/gemma_2b_repo")  # local clone
    LOCAL_SNAPSHOT_DIR = os.path.expanduser("~/gemma_2b_full")  # snapshot destination

    # ============= 1) Read token & Log in =============
    with open(HF_TOKEN_FILE, "r", encoding="utf-8") as f:
        token = f.read().strip()
    login(token=token)
    print(f"Logged in as: {whoami()['name']}")

    # ============= 2) Create private repo (no error if it exists) =============
    create_repo(repo_id=REPO_ID, private=True, token=token, exist_ok=True)
    print(f"Created or verified private repo: {REPO_ID}")

    # ============= 3) Remove local clone folder if it exists (avoid conflicts) =============
    if os.path.exists(LOCAL_REPO_DIR):
        print(f"Removing old local repo folder: {LOCAL_REPO_DIR}")
        shutil.rmtree(LOCAL_REPO_DIR)
    os.makedirs(LOCAL_REPO_DIR, exist_ok=True)

    # ============= 4) Clone the repo fresh =============
    print(f"Cloning {REPO_ID} into {LOCAL_REPO_DIR} ...")
    repo = Repository(local_dir=LOCAL_REPO_DIR, clone_from=REPO_ID, token=token)
    print("Clone complete.")

    # ============= 5) Pull remote changes (if any) =============
    try:
        print("Pulling latest from remote (rebase)...")
        repo.git_pull(rebase=True)
    except Exception as e:
        print(f"Warning: git_pull failed, ignoring. Error: {e}")

    # ============= 6) Download full snapshot of google/gemma-2-2b =============
    print(f"Downloading google/gemma-2-2b into {LOCAL_SNAPSHOT_DIR} ...")
    os.makedirs(LOCAL_SNAPSHOT_DIR, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id="google/gemma-2-2b",
        cache_dir=LOCAL_SNAPSHOT_DIR,
        local_files_only=False,
        allow_patterns=["*"],
        use_auth_token=token,
    )
    print(f"Snapshot downloaded to: {snapshot_path}")

    # ============= 7) Copy snapshot files -> local clone =============
    print("Copying snapshot files to local clone ...")
    for root, dirs, files in os.walk(snapshot_path):
        rel_path = os.path.relpath(root, snapshot_path)
        dest_subdir = os.path.join(LOCAL_REPO_DIR, rel_path)
        os.makedirs(dest_subdir, exist_ok=True)

        for filename in files:
            src = os.path.join(root, filename)
            dst = os.path.join(dest_subdir, filename)
            print(f"  Copying {src} -> {dst}")
            shutil.copy2(src, dst)

    # ============= 8) Enable large-file support & commit/push =============
    print("Enabling LFS for large files ...")
    repo.lfs_enable_largefiles()

    print("Adding/committing all files ...")
    repo.git_add(auto_lfs_track=True)
    try:
        repo.git_commit("Add gemma-2-2b full model & tokenizer")
    except Exception as e:
        if "nothing to commit" in str(e):
            print("No changes to commit, ignoring.")
        else:
            raise

    print("Attempting to push to remote ...")
    try:
        repo.git_push()
    except Exception as push_error:
        print(f"Push failed, trying a rebase pull then push again. Error:\n{push_error}")
        try:
            repo.git_pull(rebase=True)
            repo.git_push()
        except Exception as rebase_error:
            print(f"Push after rebase also failed, forcing push. Error:\n{rebase_error}")
            # Final fallback: do a forced push
            subprocess.run(
                ["git", "push", "origin", "HEAD:main", "--force"],
                check=True,
                cwd=LOCAL_REPO_DIR
            )

    print(f"Done! Check your new private repo at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
