from pydantic import BaseModel, field_validator
from fastapi import UploadFile, File
from typing import List


def parse_file_upload(
    model: UploadFile = File(...),
    weight: UploadFile = File(...),
    ckks_parms: UploadFile = File(...),
    galois_key: UploadFile = File(...),
    relin_key: UploadFile = File(...),
    pub_key: UploadFile = File(...),
    # 나머지 파일 ...
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
    model: UploadFile
    weight: UploadFile
    ckks_parms: UploadFile
    galois_key: UploadFile
    relin_key: UploadFile
    pub_key: UploadFile


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
