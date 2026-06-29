import os
import shutil


def clean_manually(cache_dir):
    count = 0
    for folder in os.listdir(cache_dir):
        count += 1
        x = None
        print(folder)
        x = input("Delete?")
        if x == "y":
            shutil.rmtree(os.path.join(cache_dir,folder))

    print(f"Total folders: {count}")


def remove_locks(cache_dir,remove=True):
    count = 0
    for folder in os.listdir(cache_dir):
        lock_path = os.path.join(cache_dir,folder,"lock")
        if os.path.exists(lock_path):
            print(f"[Cache Cleaning] Found lock file for {folder}")
            count += 1
            if remove:
                os.remove(lock_path)
        elif "solution_kernel" in folder:
            new_folder_path = os.path.join(cache_dir, folder)
            for sub_folder in os.listdir(new_folder_path):
                lock_path2 = os.path.join(new_folder_path,sub_folder,"lock")
                if os.path.exists(lock_path2):
                    print(f"[Cache Cleaning] Found lock file for {folder} & {sub_folder}")
                    count += 1
                    if remove:
                        os.remove(lock_path2) 

    print(f"Lock cleaning finished. Found: {count} locks. We have removed them: {remove}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true", help="if you want to remove the locks")
    parser.add_argument("--cache_dir", type=str, help="Cache dir for locks...")
    parser.add_argument("--experiment", type=str, default="exp_v3", help="Cache dir for locks...")
    parser.add_argument("--level", type=int, default=1, help="Cache dir for locks...")
    parser.add_argument("--trial", type=int, default=1, help="Cache dir for locks...")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    # CACHE_FOLDER="/root/.cache/torch_extensions/py311_cu128"

    args = get_args()

    CACHE_FOLDER=f"./experiments/{args.experiment}/trial_{args.trial}/kernel"


    if not args.cache_dir:
        cache_dir = CACHE_FOLDER
    else:
        cache_dir = args.cache_dir

    remove_locks(cache_dir, args.remove)