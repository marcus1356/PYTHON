from fastapi import APIRouter

from app.api.v1.endpoints import analysis, detect, submissions

api_router = APIRouter()
api_router.include_router(submissions.router)
api_router.include_router(analysis.router)
api_router.include_router(detect.router)
