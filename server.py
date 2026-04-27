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
