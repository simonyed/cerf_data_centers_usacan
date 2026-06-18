import json
from pathlib import Path

import geopandas as gpd


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  >
  <style>
    html, body {{
      margin: 0;
      height: 100%;
      font-family: Arial, sans-serif;
      background: #f5f6f8;
      color: #1f2937;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 360px 1fr;
      height: 100%;
    }}
    .panel {{
      overflow: auto;
      padding: 20px;
      background: #ffffff;
      border-right: 1px solid #d1d5db;
    }}
    .panel h1 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .panel p {{
      margin: 0 0 16px;
      line-height: 1.5;
      color: #4b5563;
    }}
    .metric {{
      margin: 0 0 12px;
      padding: 12px 14px;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #f9fafb;
    }}
    .metric strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #6b7280;
    }}
    #map {{
      height: 100%;
      width: 100%;
    }}
    .leaflet-popup-content {{
      line-height: 1.5;
    }}
    @media (max-width: 900px) {{
      .layout {{
        grid-template-columns: 1fr;
        grid-template-rows: auto 1fr;
      }}
      .panel {{
        border-right: none;
        border-bottom: 1px solid #d1d5db;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel">
      <h1>{title}</h1>
      <p>{description}</p>
      {metrics_html}
    </aside>
    <div id="map"></div>
  </div>

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const geojson = {geojson};
    const map = L.map("map", {{
      zoomControl: true
    }});

    L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }}).addTo(map);

    function numberFormat(value) {{
      if (typeof value !== "number") return value;
      return value.toLocaleString(undefined, {{ maximumFractionDigits: 4 }});
    }}

    function popupHtml(properties) {{
      const rows = [
        ["ID", properties.id],
        ["Region", properties.region],
        ["Locational Cost (M USD)", numberFormat(properties.locational_cost_million_usd)],
        ["Weighted Siting Score", numberFormat(properties.weighted_siting_score)],
        ["Interconnection Distance (km)", numberFormat(properties.interconnection_distance_km)],
        ["Electricity Cost (USD)", numberFormat(properties.electricity_cost_usd)]
      ];
      return rows.map(([label, value]) => `<strong>${{label}}:</strong> ${{value}}`).join("<br>");
    }}

    const layer = L.geoJSON(geojson, {{
      style: () => ({{
        color: "#0f766e",
        weight: 2,
        fillColor: "#14b8a6",
        fillOpacity: 0.35
      }}),
      onEachFeature: (feature, featureLayer) => {{
        featureLayer.bindPopup(popupHtml(feature.properties));
      }}
    }}).addTo(map);

    map.fitBounds(layer.getBounds(), {{ padding: [20, 20] }});
  </script>
</body>
</html>
"""


def export_geojson_properties_to_csv(geojson_path: str | Path, csv_path: str | Path | None = None) -> Path:
    geojson_path = Path(geojson_path)
    csv_path = Path(csv_path) if csv_path else geojson_path.with_suffix(".csv")

    gdf = gpd.read_file(geojson_path)
    gdf.drop(columns="geometry").to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def render_geojson_map_html(geojson_path: str | Path, html_path: str | Path | None = None) -> Path:
    geojson_path = Path(geojson_path)
    html_path = Path(html_path) if html_path else geojson_path.with_name(f"{geojson_path.stem}_map.html")

    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError(f"No features found in {geojson_path}")

    wgs84_gdf = gdf.to_crs(epsg=4326)
    geojson_obj = json.loads(wgs84_gdf.to_json())

    first = gdf.iloc[0]
    unique_regions = sorted(gdf["region"].dropna().astype(str).unique().tolist())
    if len(unique_regions) == 1:
        title = f"{unique_regions[0]} Siting Result"
        region_metric = unique_regions[0]
    else:
        title = f"{geojson_path.stem} Siting Result"
        region_metric = f"{len(unique_regions)} regions"
    description = (
        "Interactive map generated from the CERF Data Centers output GeoJSON. "
        "The geometry was reprojected from ESRI:102003 to EPSG:4326 for web display."
    )
    metrics = [
        ("Feature count", str(len(gdf))),
        ("Region", region_metric),
        ("Site ID", str(first["id"])),
        ("Locational cost (M USD)", f"{first['locational_cost_million_usd']:.4f}"),
        ("Weighted siting score", f"{first['weighted_siting_score']:.6f}"),
        ("Interconnection distance (km)", f"{first['interconnection_distance_km']:.4f}"),
    ]
    metrics_html = "\n".join(
        f'<div class="metric"><strong>{label}</strong>{value}</div>'
        for label, value in metrics
    )

    html = HTML_TEMPLATE.format(
        title=title,
        description=description,
        metrics_html=metrics_html,
        geojson=json.dumps(geojson_obj),
    )
    html_path.write_text(html, encoding="utf-8")
    return html_path


def generate_output_artifacts(geojson_path: str | Path) -> tuple[Path, Path]:
    geojson_path = Path(geojson_path)
    html_path = render_geojson_map_html(geojson_path)
    csv_path = export_geojson_properties_to_csv(geojson_path)
    return html_path, csv_path
