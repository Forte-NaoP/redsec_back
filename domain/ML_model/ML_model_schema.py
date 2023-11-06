from pydantic import BaseModel, field_validator
from fastapi import UploadFile, File
from typing import List, Optional


def parse_file_upload(
    model: UploadFile = File(None),
    weight: UploadFile = File(None),
    ckks_parms: UploadFile = File(None),
    galois_key: UploadFile = File(None),
    relin_key: UploadFile = File(None),
    pub_key: UploadFile = File(None),
):
    return FileUpload(
        model=model,
        weight=weight,
        ckks_parms=ckks_parms,
        galois_key=galois_key,
        relin_key=relin_key,
        pub_key=pub_key,
    )


class FileUpload(BaseModel):
    model: Optional[UploadFile] = None
    weight: Optional[UploadFile] = None
    ckks_parms: Optional[UploadFile] = None
    galois_key: Optional[UploadFile] = None
    relin_key: Optional[UploadFile] = None
    pub_key: Optional[UploadFile] = None

    def __getitem__(self, item):
        return getattr(self, item)


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
    models: List[ModelPublic] = []


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
    history: List[ModelHistoryPublic] = []
