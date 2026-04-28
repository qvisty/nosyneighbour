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

from nosy_neighbour import TinglysningClient, get_loan_type_info, kommune_kode, fetch_price_trend

DAWA_REVERSE_URL = "https://api.dataforsyningen.dk/adgangsadresser/reverse"

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


REJSEPLANEN_NEARBY_URL = "http://xmlopen.rejseplanen.dk/bin/rest.exe/location.nearbystops"
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
    """Return nearby public transport stops from Rejseplanen."""
    resp = requests.get(REJSEPLANEN_NEARBY_URL, params={
        "coordX": str(int(lng * 1_000_000)),
        "coordY": str(int(lat * 1_000_000)),
        "maxNo": max_results,
        "format": "json",
    }, timeout=10)
    if not resp.ok:
        raise HTTPException(status_code=502, detail="Rejseplanen API error")
    data = resp.json()
    stops_raw = data.get("LocationList", {}).get("StopLocation", [])
    if isinstance(stops_raw, dict):
        stops_raw = [stops_raw]
    stops = []
    for s in stops_raw:
        try:
            stops.append({
                "name": s.get("name", ""),
                "lat": int(s.get("y", 0)) / 1_000_000,
                "lng": int(s.get("x", 0)) / 1_000_000,
                "distance": int(s.get("distance", 0)),
                "id": s.get("id", ""),
            })
        except (ValueError, TypeError):
            continue
    return {"stops": stops}


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

    template = _jinja_env.get_template("report.html")
    html_content = template.render(
        data=data,
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
