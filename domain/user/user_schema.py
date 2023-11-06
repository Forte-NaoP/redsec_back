from pydantic import BaseModel, field_validator, EmailStr
from typing import List
from domain.ML_model.ML_model_schema import Model


class UserCreate(BaseModel):
    username: str
    password1: str
    password2: str
    email: EmailStr

    @field_validator('username', 'password1', 'password2', 'email')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

    @field_validator('password2')
    def passwords_match(cls, v, values): # values 타입 : FieldValidationInfo(config={'title': 'UserCreate'}, context=None, data={'username': 'pahkey1', 'password1': '1'}, field_name='password2')
        if 'password1' in values.data and v != values.data['password1']:
            raise ValueError('비밀번호가 일치하지 않습니다')
        return v


class Token(BaseModel):
    access_token: str
    token_type: str
    username: str


class User(BaseModel):
    id: int
    username: str
    email: str
    user_models: List[Model] = []

    class Config:
        from_attributes = True
