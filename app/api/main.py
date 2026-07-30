"""API FastAPI — bungkus jalankan_precheck (assemble.py) jadi HTTP. Tidak ada logika reasoning baru.

CLAUDE.md § Pembagian kerja: "Saya -> ... -> API." Endpoint murni transport: validasi request,
panggil pipeline reasoning, kembalikan response — semua logika reasoning tetap di app/reasoning/*.
"""

import logging

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import get_retriever
from app.api.rate_limit import cek_rate_limit
from app.reasoning.assemble import jalankan_precheck
from app.retrieval.base import Retriever
from app.schemas import L2Assessment, OutputL3

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RDTR Sleman — AI Reasoning (L3 Advisory)",
    description=(
        "Komponen AI Reasoning untuk pre-check risiko izin bangunan Kabupaten Sleman: menerima "
        "gate_hukum + impact_assessment dari back-end (L2), menghasilkan reasoning, sitasi "
        "terverifikasi, rekomendasi, rekomendasi_sistem, dan kesimpulan sebagai JSON."
    ),
    version="0.2.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reasoning", response_model=OutputL3, dependencies=[Depends(cek_rate_limit)])
def reasoning(
    request: L2Assessment,
    retriever: Retriever = Depends(get_retriever),
) -> OutputL3:
    return jalankan_precheck(request, retriever)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error di %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "Terjadi kesalahan internal. Silakan coba lagi atau hubungi admin.",
        },
    )
