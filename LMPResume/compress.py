#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import shutil
import tarfile
from pathlib import Path


def copy_items_iterative(src_path: Path, dest_path: Path, max_size_bytes: int) -> None:
    stack = [(src_path, dest_path)]
    while stack:
        current_src, current_dest = stack.pop()
        for item in current_src.iterdir():
            dest_item = current_dest / item.name
            if item.is_dir():
                dest_item.mkdir(parents=True, exist_ok=True)
                stack.append((item, dest_item))
            elif item.is_file():
                try:
                    if item.stat().st_size <= max_size_bytes:
                        shutil.copy2(item, dest_item)
                    else:
                        print(f"Skipping file (size exceeds {max_size_bytes}B): {item}")
                except OSError as e:
                    print(f"Error copying file {item}: {e}")


def copy_items_recursive(src_path: Path, dest_path: Path, max_size_bytes: int) -> None:
    for item in src_path.iterdir():
        dest_item = dest_path / item.name
        if item.is_dir():
            # Create the directory in the destination
            dest_item.mkdir(parents=True, exist_ok=True)
            # Recursively copy items in the directory
            copy_items_recursive(item, dest_item, max_size_bytes)
        elif item.is_file():
            try:
                # Copy the file if it's smaller than the maximum size
                if item.stat().st_size <= max_size_bytes:
                    shutil.copy2(item, dest_item)
                else:
                    print(f"Skipping file (size exceeds {max_size_bytes}B): {item}")
            except OSError as e:
                print(f"Error copying file {item}: {e}")


def copy_and_compress_folder_lzma(src_folder: Path, dest_folder: Path, max_size_mb: float, delete_uncompressed: bool = True) -> Path:
    """
    Recursively copy a folder to a new location, including hidden files and directories,
    excluding files bigger than max_size_mb. Then compress the copied folder into a .tar.xz
    archive using LZMA compression and delete the uncompressed copy.

    Parameters:
    src_folder (str or Path): The path to the source folder.
    dest_folder (str or Path): The path to the destination folder.
    max_size_mb (float): The maximum file size (in megabytes (rounded after conversion to bytes)) to copy.

    Returns:
    Path: The path to the compressed archive.
    """
    max_size_bytes = int(max_size_mb * 1024 * 1024)  # Convert megabytes to bytes

    # Check if the source folder exists
    if not src_folder.exists():
        raise FileNotFoundError(f"Source folder '{src_folder}' does not exist.")

    # If the destination folder exists, remove it to ensure a clean copy
    if dest_folder.exists():
        # shutil.rmtree(dest_folder)
        raise RuntimeError("Destination folder exists")

    # Start copying from the root directory
    dest_folder.mkdir(parents=True, exist_ok=True)
    copy_items_iterative(src_folder, dest_folder, max_size_bytes)

    # Compress the copied folder using LZMA compression
    archive_name = dest_folder.with_suffix('.tar.xz')
    with tarfile.open(archive_name, 'w:xz') as tar:
        tar.add(dest_folder, arcname=dest_folder.name)

    # Delete the uncompressed destination folder
    if delete_uncompressed:
        shutil.rmtree(dest_folder)

    print(f"Folder compressed into archive: {archive_name}")
    return archive_name


if __name__ == "__main__":
    pass
