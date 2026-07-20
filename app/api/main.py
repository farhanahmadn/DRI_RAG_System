"""API FastAPI — bungkus jalankan_precheck (assemble.py) jadi HTTP. Tidak ada logika reasoning baru.

CLAUDE.md § Pembagian kerja: "Saya -> ... -> API." Endpoint murni transport: validasi request,
panggil pipeline reasoning, kembalikan response — semua logika reasoning tetap di app/reasoning/*.
"""

import logging

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import get_retriever
from app.reasoning.assemble import jalankan_precheck
from app.retrieval.base import Retriever
from app.schemas import JejakAturanRequest, OutputPreCheck

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RDTR Sleman — AI Reasoning",
    description=(
        "Komponen AI Reasoning untuk pre-check risiko izin bangunan Kabupaten Sleman: menerima "
        "jejak aturan rule-based, menghasilkan reasoning, sitasi terverifikasi, rekomendasi, dan "
        "kesimpulan sebagai JSON."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reasoning", response_model=OutputPreCheck)
def reasoning(
    request: JejakAturanRequest,
    retriever: Retriever = Depends(get_retriever),
) -> OutputPreCheck:
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
