from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCE_COLUMNS = [
    "land_building_cost_usd",
    "equipment_capex_usd",
    "real_property_taxes_usd",
    "personal_property_taxes_usd",
    "equipment_sales_tax_usd",
    "interconnection cost",
]


def recompute_total_capex_cost(df: pd.DataFrame) -> pd.Series:
    missing = [column for column in SOURCE_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少用于计算 total_capex_cost 的列: {missing}")

    return df[SOURCE_COLUMNS].fillna(0).sum(axis=1)


def build_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}_recomputed{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据其他成本列重算 total_capex_cost。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("usa_can_merged.csv"),
        help="输入 CSV 路径，默认是当前目录下的 usa_can_merged.csv。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径，默认生成 *_recomputed.csv。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="直接覆盖输入文件。",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="校验原 total_capex_cost 时使用的误差阈值。",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = input_path if args.overwrite else build_output_path(input_path, args.output)

    df = pd.read_csv(input_path)
    recomputed = recompute_total_capex_cost(df)

    if "total_capex_cost" in df.columns:
        diff = (df["total_capex_cost"] - recomputed).abs()
        mismatch_count = int((diff > args.tolerance).sum())
        print(f"rows={len(df)}")
        print(f"max_abs_diff={diff.max()}")
        print(f"mismatch_count_gt_tolerance={mismatch_count}")
    else:
        print("输入文件没有 total_capex_cost 列，将新建该列。")

    df["total_capex_cost"] = recomputed
    df.to_csv(output_path, index=False)
    print(f"written={output_path}")


if __name__ == "__main__":
    main()
