from pydantic import BaseModel


class BusinessUseCase(BaseModel):
    scenario: str
