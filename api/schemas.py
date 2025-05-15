from pydantic import BaseModel

class PricesRequest(BaseModel):
    symbol: str
    seq_length: int = 60

class PredictionResponse(BaseModel):
    next_price: float
