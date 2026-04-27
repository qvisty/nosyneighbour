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

from nosy_neighbour import TinglysningClient, get_loan_type_info

DAWA_REVERSE_URL = "https://api.dataforsyningen.dk/adgangsadresser/reverse"
SVUR_URL = "https://services.datafordeler.dk/SVUR/SVURStamdata/1/REST/Vurderingsejendom"
REJSEPLANEN_NEARBY_URL = "http://xmlopen.rejseplanen.dk/bin/rest.exe/location.nearbystops"
DMI_CLIMATE_URL = "https://dmigw.govcloud.dk/v2/climateData/collections/municipalityValue/items"

DMI_API_KEY = os.environ.get("DMI_API_KEY", "")
DATAFORDELER_USER = os.environ.get("DATAFORDELER_USER", "")
DATAFORDELER_PASS = os.environ.get("DATAFORDELER_PASS", "")

CLIMATE_PARAMS = [
    "mean_temp", "mean_daily_max_temp", "mean_daily_min_temp",
    "acc_precip", "mean_wind_speed", "bright_sunshine",
]

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


def _resolve_with_coords(q: str) -> tuple[str, str, str, float | None, float | None]:
    """Resolve freeform address to (postnr, vejnavn, husnr, lat, lng).

    Lat/lng are returned as None if DAWA didn't include coordinates.
    """
    results = _client.autocomplete_address(q)
    for r in results:
        d = r.get("data", {})
        if d.get("postnr") and d.get("vejnavn") and d.get("husnr"):
            return d["postnr"], d["vejnavn"], d["husnr"], d.get("y"), d.get("x")
    raise RuntimeError(f"Could not resolve address: {q!r}")


def _parse_kr(s) -> int:
    """'1.200.000 kr.' -> 1200000."""
    if not s:
        return 0
    digits = "".join(c for c in str(s) if c.isdigit())
    return int(digits) if digits else 0


def _loan_age_months(alias: str | None) -> int | None:
    if not alias:
        return None
    try:
        date_part = alias.split("-")[0]
        d, m, y = (int(x) for x in date_part.split("."))
        reg = datetime(y, m, d)
    except (ValueError, IndexError):
        return None
    now = datetime.now()
    return (now.year - reg.year) * 12 + (now.month - reg.month)


def _outstanding_balance(principal: float, annual_rate_pct: float,
                         months_elapsed: int | None, profile: str) -> float:
    """Estimate outstanding balance under one of three amortization profiles."""
    if months_elapsed is None or months_elapsed <= 0:
        return principal
    r = annual_rate_pct / 100 / 12

    def annuity_remaining(P: float, r: float, n: int, k: int) -> float:
        k = min(k, n)
        if k >= n:
            return 0
        if r == 0:
            return P * (n - k) / n
        return P * ((1 + r) ** n - (1 + r) ** k) / ((1 + r) ** n - 1)

    if profile == "interest_only":
        return principal
    if profile == "repayment":
        return annuity_remaining(principal, r, 360, months_elapsed)
    if profile == "deferred_10y":
        if months_elapsed <= 120:
            return principal
        return annuity_remaining(principal, r, 240, months_elapsed - 120)
    return principal


def _equity_estimate(data: dict, profile: str = "deferred_10y") -> dict | None:
    """Return {total_debt, frivaerdi, frivaerdi_pct, profile} or None."""
    v = data.get("vurdering")
    if not v or not v.get("ejendomsvaerdi"):
        return None
    realkredit_types = ("Realkreditpantebrev", "Afgiftspantebrev")
    total = 0.0
    for h in data.get("haeftelser") or []:
        principal = _parse_kr(h.get("hovedstol"))
        rate = float(h.get("rente") or 0)
        months = _loan_age_months(h.get("alias"))
        if h.get("haeftelsestype") in realkredit_types and months is not None and rate > 0:
            total += _outstanding_balance(principal, rate, months, profile)
        else:
            total += principal
    ejvaerdi = v["ejendomsvaerdi"]
    frivaerdi = ejvaerdi - total
    return {
        "total_debt": round(total),
        "frivaerdi": round(frivaerdi),
        "frivaerdi_pct": round(frivaerdi / ejvaerdi * 100) if ejvaerdi > 0 else None,
        "profile": profile,
    }


def _fetch_valuations(tingbog: dict) -> dict:
    """Return {adresse, kommune, history: [{year, ejendomsvaerdi, grundvaerdi}]}.

    Always includes the current valuation from the tingbog. Historical values
    are fetched from SVUR (Datafordeler) when DATAFORDELER_USER/PASS is set.
    """
    kommune = (tingbog.get("vurdering") or {}).get("kommune", "")
    matrikler = tingbog.get("matrikler") or []
    history: list[dict] = []

    if matrikler and DATAFORDELER_USER and DATAFORDELER_PASS:
        m = matrikler[0]
        ejerlav_kode = m.get("ejerlavskode")
        matrikel_nr = m.get("matrikelnummer")
        if ejerlav_kode and matrikel_nr:
            try:
                resp = requests.get(SVUR_URL, params={
                    "EjerlavId": ejerlav_kode,
                    "MatrikelNr": matrikel_nr,
                    "format": "json",
                    "username": DATAFORDELER_USER,
                    "password": DATAFORDELER_PASS,
                }, timeout=10)
                if resp.ok:
                    data = resp.json()
                    items = data if isinstance(data, list) else data.get("features", data.get("items", []))
                    for item in items:
                        props = item.get("properties", item) if isinstance(item, dict) else {}
                        year = props.get("vurderingsaar") or props.get("vurderingAar")
                        ejvaerdi = props.get("ejendomsvaerdi") or props.get("ejendomsVaerdi")
                        grundvaerdi = props.get("grundvaerdi") or props.get("grundVaerdi")
                        if year and (ejvaerdi or grundvaerdi):
                            history.append({
                                "year": int(year),
                                "ejendomsvaerdi": int(ejvaerdi) if ejvaerdi else None,
                                "grundvaerdi": int(grundvaerdi) if grundvaerdi else None,
                            })
            except Exception:
                pass

    current = tingbog.get("vurdering")
    if current:
        year_str = current.get("vurderingsdato", "")[:4]
        if year_str:
            current_year = int(year_str)
            if not any(h["year"] == current_year for h in history):
                history.append({
                    "year": current_year,
                    "ejendomsvaerdi": current.get("ejendomsvaerdi"),
                    "grundvaerdi": current.get("grundvaerdi"),
                })

    history.sort(key=lambda x: x["year"])
    return {"adresse": tingbog.get("adresse"), "kommune": kommune, "history": history}


def _fetch_climate(lat: float, lng: float) -> dict:
    """Return {kommune, parameters: {param_id: [{year, value}]}}."""
    resp = requests.get(DAWA_REVERSE_URL, params={"x": lng, "y": lat})
    if not resp.ok:
        raise HTTPException(status_code=404, detail="Could not resolve location")
    kommune = resp.json().get("kommune") or {}
    kommune_kode = kommune.get("kode")
    kommune_navn = kommune.get("navn", "")
    if not kommune_kode:
        raise HTTPException(status_code=404, detail="Could not determine municipality")

    base_params = {
        "municipalityId": kommune_kode,
        "timeResolution": "year",
        "limit": 30,
        "api-key": DMI_API_KEY,
    }
    out: dict = {"kommune": kommune_navn, "parameters": {}}
    for param_id in CLIMATE_PARAMS:
        try:
            r = requests.get(DMI_CLIMATE_URL, params={**base_params, "parameterId": param_id}, timeout=10)
            if not r.ok:
                continue
            values = []
            for f in r.json().get("features", []):
                props = f.get("properties", {})
                val = props.get("value")
                year_str = (props.get("from") or "")[:4]
                if val is not None and year_str:
                    values.append({"year": int(year_str), "value": round(val, 1)})
            if values:
                values.sort(key=lambda x: x["year"])
                out["parameters"][param_id] = values
        except Exception:
            continue
    return out


def _fetch_transport(lat: float, lng: float, max_results: int = 8) -> dict:
    """Return {stops: [{name, lat, lng, distance, id}]} from Rejseplanen."""
    try:
        r = requests.get(REJSEPLANEN_NEARBY_URL, params={
            "coordX": str(int(lng * 1_000_000)),
            "coordY": str(int(lat * 1_000_000)),
            "maxNo": max_results,
            "format": "json",
        }, timeout=10)
    except requests.RequestException:
        return {"stops": []}
    if not r.ok:
        return {"stops": []}
    raw = r.json().get("LocationList", {}).get("StopLocation", [])
    if isinstance(raw, dict):
        raw = [raw]
    stops = []
    for s in raw:
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
    return _fetch_valuations(tingbog)


@app.get("/api/climate")
def climate(lat: float = Query(...), lng: float = Query(...)):
    """Return climate normals for the nearest municipality from DMI."""
    return _fetch_climate(lat, lng)


@app.get("/api/transport")
def transport(lat: float = Query(...), lng: float = Query(...), max_results: int = Query(8)):
    """Return nearby public transport stops from Rejseplanen."""
    return _fetch_transport(lat, lng, max_results)


_CLIMATE_LABELS = {
    "mean_temp": ("Gns. temperatur", "°C"),
    "mean_daily_max_temp": ("Gns. daglig maks.", "°C"),
    "mean_daily_min_temp": ("Gns. daglig min.", "°C"),
    "acc_precip": ("Nedbør (år)", "mm"),
    "mean_wind_speed": ("Gns. vindstyrke", "m/s"),
    "bright_sunshine": ("Solskinstimer (år)", "timer"),
}


@app.get("/api/report")
def report(q: str = Query(...)):
    try:
        postnummer, vejnavn, husnummer, lat, lng = _resolve_with_coords(q)
        tingbog = _client.lookup_address(postnummer, vejnavn, husnummer)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if tingbog is None:
        raise HTTPException(status_code=404, detail="No property data found")
    data = _annotate_loan_types(tingbog)

    valuations_data = _fetch_valuations(tingbog)
    climate_data = None
    transport_data = None
    if lat is not None and lng is not None:
        try:
            climate_data = _fetch_climate(lat, lng)
        except HTTPException:
            climate_data = None
        transport_data = _fetch_transport(lat, lng)

    climate_latest = None
    if climate_data and climate_data.get("parameters"):
        climate_latest = []
        for key, values in climate_data["parameters"].items():
            if not values or key not in _CLIMATE_LABELS:
                continue
            last = values[-1]
            label, unit = _CLIMATE_LABELS[key]
            climate_latest.append({
                "label": label, "unit": unit,
                "value": last["value"], "year": last["year"],
            })

    template = _jinja_env.get_template("report.html")
    html_content = template.render(
        data=data,
        equity=_equity_estimate(data),
        valuations=valuations_data,
        climate=climate_data,
        climate_latest=climate_latest,
        transport=transport_data,
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
