import sys
from pathlib import Path

from cerf_data_centers.postprocess_output import render_geojson_map_html


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/render_geojson_map.py <input_geojson> <output_html>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    render_geojson_map_html(input_path, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
