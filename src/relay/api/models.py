"""GET /v1/models and /v1/models/{id}: the public model names from the config."""

from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from relay.api.errors import model_not_found
from relay.api.schemas import ErrorResponse

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
    # Answered from config, not by asking backends: instant, stable, and filterable
    # per tenant later.
    return ModelList(data=[_model(request, m) for m in request.app.state.config.models])


@router.get(
    "/v1/models/{model_id:path}",
    response_model=Model,
    responses={404: {"model": ErrorResponse}},
)
async def retrieve_model(model_id: str, request: Request) -> Model | Response:
    if model_id not in request.app.state.config.models:
        return model_not_found(model_id)
    return _model(request, model_id)
