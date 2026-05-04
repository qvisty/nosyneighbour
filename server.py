"""
nosy-neighbour web server.

Serves a map-based UI, a JSON REST API, and an MCP server at POST /mcp.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from jinja2 import Environment, FileSystemLoader
from mcp.server.fastmcp import FastMCP
import uvicorn
from weasyprint import HTML

from nosy_neighbour import TinglysningClient, get_loan_type_info, kommune_kode, fetch_price_trend, fetch_dst_demographics, fetch_bbr_data

DAWA_REVERSE_URL = "https://api.dataforsyningen.dk/adgangsadresser/reverse"
DATAFORSYNINGEN_TOKEN = os.environ.get("DATAFORSYNINGEN_TOKEN", "")

_templates_dir = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(str(_templates_dir)), autoescape=True)

log = logging.getLogger(__name__)

_client = TinglysningClient()

with open("templates/index.html") as f:
    _index_html = f.read()


def _annotate_loan_types(tingbog: dict) -> dict:
    for h in tingbog.get("haeftelser") or []:
        rente = float(h.get("rente") or 0)
        if (h.get("fastvariabel") == "variabel"
                and h.get("haeftelsestype") in ("Realkreditpantebrev", "Afgiftspantebrev")
                and rente > 0):
            h["loan_type_info"] = get_loan_type_info(rente, alias=h.get("alias"))
    return tingbog


# ── MCP server ────────────────────────────────────────────────────────────────
mcp_server = FastMCP("nosy-neighbour", stateless_http=True, json_response=True)


@mcp_server.tool()
def lookup_property(address: str) -> dict:
    """Look up Danish property records from tinglysning.dk.

    Given a freeform Danish address, returns owners (ejere), official
    valuation (vurdering) with equity estimate, mortgages and liens
    (hæftelser) with loan-type estimation for variable-rate realkreditlån,
    and easements (servitutter).
    """
    try:
        postnummer, vejnavn, husnummer = _client.resolve_address(address)
        tingbog = _client.lookup_address(postnummer, vejnavn, husnummer)
    except RuntimeError as e:
        return {"error": str(e)}
    if tingbog is None:
        return {"error": "No property data found"}
    return _annotate_loan_types(tingbog)


# ── FastAPI app ───────────────────────────────────────────────────────────────
_mcp_asgi = mcp_server.streamable_http_app()  # lazily initialises session_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(title="nosy-neighbour", lifespan=lifespan)


@app.get("/api/autocomplete")
def autocomplete(q: str = Query(...)):
    results = _client.autocomplete_address(q)
    return [
        {
            "label": r["forslagstekst"],
            "postnr": d["postnr"],
            "vejnavn": d["vejnavn"],
            "husnr": d["husnr"],
            "lat": d["y"],
            "lng": d["x"],
            "adgangsadresse_id": d.get("id"),
        }
        for r in results
        if (d := r.get("data", {})) and d.get("postnr") and d.get("vejnavn") and d.get("husnr")
    ]


@app.get("/api/reverse")
def reverse(lat: float = Query(...), lng: float = Query(...)):
    resp = requests.get(DAWA_REVERSE_URL, params={"x": lng, "y": lat})
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="No address found at this location")
    resp.raise_for_status()
    d = resp.json()
    return {
        "label": d["adressebetegnelse"],
        "postnr": d["postnummer"]["nr"],
        "vejnavn": d["vejstykke"]["navn"],
        "husnr": d["husnr"],
        "lat": d["adgangspunkt"]["koordinater"][1],
        "lng": d["adgangspunkt"]["koordinater"][0],
        "adgangsadresse_id": d.get("id"),
    }


@app.get("/api/lookup")
def lookup(q: str = Query(...)):
    try:
        postnummer, vejnavn, husnummer = _client.resolve_address(q)
        tingbog = _client.lookup_address(postnummer, vejnavn, husnummer)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if tingbog is None:
        raise HTTPException(status_code=404, detail="No property data found")
    return _annotate_loan_types(tingbog)


@app.get("/api/valuations")
def valuations(q: str = Query(...)):
    """Return historical property valuations as a time series."""
    try:
        postnummer, vejnavn, husnummer = _client.resolve_address(q)
        tingbog = _client.lookup_address(postnummer, vejnavn, husnummer)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if tingbog is None:
        raise HTTPException(status_code=404, detail="No property data found")

    vurdering = tingbog.get("vurdering") or {}
    kommune = vurdering.get("kommune", "")
    if not kommune:
        raise HTTPException(status_code=404, detail="No valuation data for this property")

    kode = kommune_kode(kommune)
    if not kode:
        raise HTTPException(status_code=404, detail=f"Unknown municipality: {kommune}")

    # Use DST EJ67 regional price index to estimate historical valuations
    trend = fetch_price_trend(
        kommunekode=kode,
        ejendomstype=tingbog.get("ejendomstype"),
        current_ejendomsvaerdi=vurdering.get("ejendomsvaerdi"),
        current_grundvaerdi=vurdering.get("grundvaerdi"),
        vurderingsdato=vurdering.get("vurderingsdato"),
    )

    # Build history array compatible with the existing frontend chart
    history = []
    if trend and trend.get("vurderinger"):
        for v in trend["vurderinger"]:
            history.append({
                "year": int(v["aar"]),
                "ejendomsvaerdi": v.get("ejendomsvaerdi_est"),
                "grundvaerdi": v.get("grundvaerdi_est"),
            })

    # Always include the actual current valuation (replaces any estimate for that year)
    current_year_str = vurdering.get("vurderingsdato", "")[:4]
    if current_year_str:
        current_year = int(current_year_str)
        history = [h for h in history if h["year"] != current_year]
        history.append({
            "year": current_year,
            "ejendomsvaerdi": vurdering.get("ejendomsvaerdi"),
            "grundvaerdi": vurdering.get("grundvaerdi"),
        })

    history.sort(key=lambda x: x["year"])

    return {
        "adresse": tingbog.get("adresse"),
        "kommune": kommune,
        "kilde": trend["kilde"] if trend else None,
        "region": trend["region"] if trend else None,
        "history": history,
    }


REJSEPLANEN_BASE_URL = "https://www.rejseplanen.dk/api"
REJSEPLANEN_ACCESS_ID = os.environ.get("REJSEPLANEN_ACCESS_ID", "")
DMI_CLIMATE_URL = "https://dmigw.govcloud.dk/v2/climateData/collections/municipalityValue/items"
DMI_API_KEY = os.environ.get("DMI_API_KEY", "")


@app.get("/api/climate")
def climate(lat: float = Query(...), lng: float = Query(...)):
    """Return climate normals for the nearest municipality from DMI."""
    # First resolve lat/lng to municipality via DAWA
    resp = requests.get(DAWA_REVERSE_URL, params={"x": lng, "y": lat})
    if not resp.ok:
        raise HTTPException(status_code=404, detail="Could not resolve location")
    kommune_kode = resp.json().get("kommune", {}).get("kode")
    kommune_navn = resp.json().get("kommune", {}).get("navn", "")
    if not kommune_kode:
        raise HTTPException(status_code=404, detail="Could not determine municipality")

    # Fetch climate normals from DMI
    params = {
        "municipalityId": kommune_kode,
        "timeResolution": "year",
        "limit": 30,
        "api-key": DMI_API_KEY,
    }
    climate_data = {"kommune": kommune_navn, "parameters": {}}

    for param_id in ["mean_temp", "mean_daily_max_temp", "mean_daily_min_temp",
                      "acc_precip", "mean_wind_speed", "bright_sunshine"]:
        try:
            resp = requests.get(DMI_CLIMATE_URL, params={**params, "parameterId": param_id}, timeout=10)
            if resp.ok:
                features = resp.json().get("features", [])
                values = []
                for f in features:
                    props = f.get("properties", {})
                    val = props.get("value")
                    time_str = props.get("from", "")[:4]
                    if val is not None and time_str:
                        values.append({"year": int(time_str), "value": round(val, 1)})
                if values:
                    values.sort(key=lambda x: x["year"])
                    climate_data["parameters"][param_id] = values
        except Exception:
            continue

    return climate_data


@app.get("/api/transport")
def transport(lat: float = Query(...), lng: float = Query(...), max_results: int = Query(8)):
    """Return nearby public transport stops with departures from Rejseplanen API 2.0."""
    if not REJSEPLANEN_ACCESS_ID:
        raise HTTPException(status_code=503, detail="REJSEPLANEN_ACCESS_ID not configured")
    # Nearby stops
    resp = requests.get(f"{REJSEPLANEN_BASE_URL}/location.nearbystops", params={
        "accessId": REJSEPLANEN_ACCESS_ID,
        "originCoordLat": lat,
        "originCoordLong": lng,
        "r": 1000,
        "maxNo": max_results,
        "format": "json",
    }, timeout=10)
    if not resp.ok:
        raise HTTPException(status_code=502, detail="Rejseplanen API error")
    data = resp.json()
    if "errorCode" in data:
        raise HTTPException(status_code=502, detail=data.get("errorText", "Rejseplanen error"))
    stops_raw = data.get("stopLocationOrCoordLocation", [])
    stops = []
    for item in stops_raw:
        s = item.get("StopLocation", item)
        try:
            stops.append({
                "name": s.get("name", ""),
                "lat": float(s.get("lat", 0)),
                "lng": float(s.get("lon", 0)),
                "distance": int(s.get("dist", s.get("distance", 0))),
                "id": s.get("extId", s.get("id", "")),
            })
        except (ValueError, TypeError):
            continue
    # Fetch departures for the nearest stop
    departures = []
    if stops:
        nearest_id = stops[0]["id"]
        try:
            dep_resp = requests.get(f"{REJSEPLANEN_BASE_URL}/departureBoard", params={
                "accessId": REJSEPLANEN_ACCESS_ID,
                "id": nearest_id,
                "duration": 60,
                "maxJourneys": 8,
                "format": "json",
            }, timeout=10)
            if dep_resp.ok:
                dep_data = dep_resp.json()
                for d in dep_data.get("Departure", []):
                    departures.append({
                        "name": d.get("name", "").strip(),
                        "direction": d.get("direction", ""),
                        "time": d.get("time", ""),
                        "date": d.get("date", ""),
                        "track": d.get("track", ""),
                        "stop": d.get("stop", ""),
                    })
        except Exception:
            pass
    return {"stops": stops, "departures": departures}


def _fetch_aerial_photo(lat: float, lng: float, width: int = 600, height: int = 300) -> str | None:
    """Fetch an aerial photo from Dataforsyningen WMS (requires DATAFORSYNINGEN_TOKEN)."""
    import base64 as b64
    import math

    if not DATAFORSYNINGEN_TOKEN:
        return None

    # Convert WGS84 lat/lng to EPSG:25832 (UTM zone 32N) used by Danish services
    lat_rad = math.radians(lat)
    lng_rad = math.radians(lng)
    # UTM zone 32N central meridian = 9°E
    lng0 = math.radians(9.0)
    k0 = 0.9996
    a = 6378137.0
    f = 1 / 298.257223563
    e = math.sqrt(2 * f - f * f)
    e2 = e * e
    ep2 = e2 / (1 - e2)
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    T = math.tan(lat_rad) ** 2
    C = ep2 * math.cos(lat_rad) ** 2
    A = math.cos(lat_rad) * (lng_rad - lng0)
    M = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e2**3 / 3072) * math.sin(6 * lat_rad)
    )
    easting = 500000 + k0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18 * T + T**2 + 72 * C - 58 * ep2) * A**5 / 120)
    northing = k0 * (M + N * math.tan(lat_rad) * (A**2 / 2 + (5 - T + 9 * C + 4 * C**2) * A**4 / 24 + (61 - 58 * T + T**2 + 600 * C - 330 * ep2) * A**6 / 720))

    # ~0.3m/pixel at this scale gives a good neighbourhood view
    half_w = width * 0.3
    half_h = height * 0.3
    bbox = f"{easting - half_w},{northing - half_h},{easting + half_w},{northing + half_h}"

    url = "https://api.dataforsyningen.dk/orto_foraar_DAF"
    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": "orto_foraar",
        "styles": "",
        "srs": "EPSG:25832",
        "bbox": bbox,
        "width": width,
        "height": height,
        "format": "image/png",
        "token": DATAFORSYNINGEN_TOKEN,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.ok and resp.headers.get("content-type", "").startswith("image/"):
            return b64.b64encode(resp.content).decode()
    except Exception:
        pass
    return None


def _fetch_osm_map(lat: float, lng: float, width: int = 600, height: int = 300) -> str | None:
    """Fallback: fetch a static map from OpenStreetMap tiles (no token required)."""
    import base64 as b64
    import io
    import math

    from PIL import Image as PILImage

    zoom = 16
    tile_size = 256
    n = 2 ** zoom
    x_tile = (lng + 180.0) / 360.0 * n
    y_tile = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    tiles_x = math.ceil(width / tile_size) + 1
    tiles_y = math.ceil(height / tile_size) + 1
    center_tx = int(x_tile)
    center_ty = int(y_tile)
    px_offset = int((x_tile - center_tx) * tile_size)
    py_offset = int((y_tile - center_ty) * tile_size)
    composite = PILImage.new("RGB", (tiles_x * tile_size, tiles_y * tile_size))
    start_tx = center_tx - tiles_x // 2
    start_ty = center_ty - tiles_y // 2

    try:
        for dx in range(tiles_x):
            for dy in range(tiles_y):
                tx = (start_tx + dx) % n
                ty = start_ty + dy
                if ty < 0 or ty >= n:
                    continue
                url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
                resp = requests.get(url, timeout=10, headers={"User-Agent": "NosyneighbourPDF/1.0"})
                if resp.ok:
                    tile_img = PILImage.open(io.BytesIO(resp.content))
                    composite.paste(tile_img, (dx * tile_size, dy * tile_size))
        crop_x = (tiles_x // 2) * tile_size + px_offset - width // 2
        crop_y = (tiles_y // 2) * tile_size + py_offset - height // 2
        result = composite.crop((crop_x, crop_y, crop_x + width, crop_y + height))
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return b64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _fetch_static_map(lat: float, lng: float, width: int = 600, height: int = 300) -> str | None:
    """Fetch aerial photo from Dataforsyningen if token is set, else fall back to OSM."""
    return _fetch_aerial_photo(lat, lng, width, height) or _fetch_osm_map(lat, lng, width, height)


@app.get("/api/report")
def report(q: str = Query(...)):
    try:
        postnummer, vejnavn, husnummer = _client.resolve_address(q)
        tingbog = _client.lookup_address(postnummer, vejnavn, husnummer)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if tingbog is None:
        raise HTTPException(status_code=404, detail="No property data found")
    data = _annotate_loan_types(tingbog)

    # Resolve coordinates for map and get adgangsadresse_id for BBR
    map_base64 = None
    bbr_data = None
    try:
        auto_results = _client.autocomplete_address(q)
        if auto_results:
            addr_data = auto_results[0].get("data", {})
            lat = addr_data.get("y")
            lng = addr_data.get("x")
            if lat and lng:
                map_base64 = _fetch_static_map(float(lat), float(lng))
            adgangsadresse_id = addr_data.get("id")
            if adgangsadresse_id:
                bbr_data = fetch_bbr_data(adgangsadresse_id)
    except Exception:
        pass

    # Fetch neighbourhood statistics
    demographics = None
    price_trend = None
    vurdering = data.get("vurdering") or {}
    kommune = vurdering.get("kommune", "")
    if kommune:
        kode = kommune_kode(kommune)
        if kode:
            try:
                demographics = fetch_dst_demographics(kode)
            except Exception:
                pass
            try:
                price_trend = fetch_price_trend(
                    kommunekode=kode,
                    ejendomstype=data.get("ejendomstype"),
                    current_ejendomsvaerdi=vurdering.get("ejendomsvaerdi"),
                    current_grundvaerdi=vurdering.get("grundvaerdi"),
                    vurderingsdato=vurdering.get("vurderingsdato"),
                )
            except Exception:
                pass

    template = _jinja_env.get_template("report.html")
    html_content = template.render(
        data=data,
        map_base64=map_base64,
        bbr_data=bbr_data,
        demographics=demographics,
        price_trend=price_trend,
        generated_at=datetime.now().strftime("%d-%m-%Y %H:%M"),
    )
    pdf_bytes = HTML(string=html_content).write_pdf()

    address_slug = data.get("adresse", "ejendom").replace(" ", "_").replace(",", "")
    filename = f"rapport_{address_slug}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/neighbourhood")
def neighbourhood(kommune: str = Query(..., description="Municipality name from vurdering")):
    """Return socioeconomic profile for a municipality from DST."""
    kode = kommune_kode(kommune)
    if not kode:
        raise HTTPException(status_code=404, detail=f"Unknown municipality: {kommune}")
    data = fetch_dst_demographics(kode)
    if data is None:
        raise HTTPException(status_code=502, detail="Could not fetch DST data")
    data["kommune"] = kommune
    data["kommunekode"] = kode
    return data


@app.get("/api/bbr")
def bbr(q: str = Query(...)):
    """Return BBR building data for an address."""
    try:
        results = _client.autocomplete_address(q)
    except Exception:
        raise HTTPException(status_code=502, detail="Could not resolve address")
    if not results:
        raise HTTPException(status_code=404, detail="Address not found")
    addr_data = results[0].get("data", {})
    adgangsadresse_id = addr_data.get("id")
    if not adgangsadresse_id:
        raise HTTPException(status_code=404, detail="No adgangsadresse id found")
    data = fetch_bbr_data(adgangsadresse_id)
    if data is None:
        raise HTTPException(status_code=502, detail="Could not fetch BBR data")
    return data


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=_index_html)


# Mount MCP last so FastAPI routes take priority when matching paths.
# streamable_http_app() registers its handler at /mcp inside the sub-app;
# mounting the sub-app at / keeps the final endpoint at POST /mcp.
app.mount("/", _mcp_asgi)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_certfile=os.environ.get("SSL_CERTFILE"),
        ssl_keyfile=os.environ.get("SSL_KEYFILE"),
    )
