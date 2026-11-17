"""Verify a saved split and optionally verify DPO prompt membership."""

import argparse

from scripts.data_splits import DEFAULT_SPLIT_DIR, load_verified_splits, read_jsonl, validate_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--dpo-path", help="若指定，检查所有 DPO 问题都属于 train")
    args = parser.parse_args()
    splits, _ = load_verified_splits(args.split_dir)
    dpo = read_jsonl(args.dpo_path) if args.dpo_path else None
    counts = validate_splits(splits, dpo)
    print("问题组数量:", counts)
    print("检查通过：train/val/test 交集均为 0，数据哈希与 manifest 一致。")
    if dpo is not None:
        print("DPO 检查通过：chosen/rejected 问题一致，全部属于 train。")


if __name__ == "__main__":
    main()
