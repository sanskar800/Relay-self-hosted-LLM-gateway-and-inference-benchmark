"""GET /v1/models and /v1/models/{id}: the models the calling tenant may use."""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from relay.api.schemas import ErrorResponse
from relay.auth.policy import model_not_found_error, visible_models

router = APIRouter()


class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]


def _model(request: Request, model_id: str) -> Model:
    cfg = request.app.state.config.models[model_id]
    return Model(id=model_id, created=request.app.state.started_at, owned_by=cfg.owned_by)


@router.get("/v1/models")
async def list_models(request: Request) -> ModelList:
    # Answered from config and the tenant's allow-list, not by asking backends.
    allowed = visible_models(request.state.principal, request.app.state.config)
    return ModelList(data=[_model(request, m) for m in allowed])


@router.get(
    "/v1/models/{model_id:path}",
    response_model=Model,
    responses={404: {"model": ErrorResponse}},
)
async def retrieve_model(model_id: str, request: Request) -> Model:
    # A model the tenant may not use is reported as not found (same as OpenAI).
    if model_id not in visible_models(request.state.principal, request.app.state.config):
        raise model_not_found_error(model_id)
    return _model(request, model_id)
