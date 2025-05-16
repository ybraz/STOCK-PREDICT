from typing import Optional
from pydantic import BaseModel


class PredictRequest(BaseModel):
    symbol: str
    seq_length: Optional[int] = 180
    open_price: Optional[float] = None


class PredictResponse(BaseModel):
    next_price: float
    expected_return_pct: float