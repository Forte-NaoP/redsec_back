from pydantic import BaseModel, field_validator


class FileUpload(BaseModel):
    name: str


class Model(BaseModel):
    id: int
    ML_model_name: str
    ML_model_uuid: str

    class Config:
        from_attributes = True


class ModelPublic(BaseModel):
    ML_model_name: str
    ML_model_uuid: str


class ModelCreate(BaseModel):
    name: str

    @field_validator('name')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v


class ModelList(BaseModel):
    total: int = 0
    models: list[ModelPublic] = []


class ModelHistory(BaseModel):
    id: int
    description: str
    file_name: str
    pending: bool

    class Config:
        from_attributes = True

class ModelHistoryPublic(BaseModel):
    description: str
    file_name: str
    pending: bool

class ModelHistoryList(BaseModel):
    total: int = 0
    history: list[ModelHistoryPublic] = []
