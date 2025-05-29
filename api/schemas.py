from typing import Optional
from pydantic import BaseModel

class PredictRequest(BaseModel):
    """
    Modelo para requisição de previsão.

    Atributos:
        symbol (str): Ticker do ativo (ex: 'PETR4.SA').
        seq_length (Optional[int]): Tamanho da janela de sequência de entrada. Default = 180.
        open_price (Optional[float]): Preço de abertura do dia para referência. Default = None.
    """
    symbol: str
    seq_length: Optional[int] = 180
    open_price: Optional[float] = None

class PredictResponse(BaseModel):
    """
    Modelo para resposta da previsão.

    Atributos:
        next_price (float): Preço previsto de fechamento (ou próximo preço, conforme implementação).
        expected_return_pct (float): Retorno percentual esperado em relação ao preço de abertura informado.
    """
    next_price: float
    expected_return_pct: float

class TrainRequest(BaseModel):
    symbol: str
    start_date: Optional[str] = "2005-01-01"
    end_date: Optional[str] = None 

class TrainResponse(BaseModel):
    message: str 
    model_path: str
    scaler_x_path: str 
    scaler_y_path: str  