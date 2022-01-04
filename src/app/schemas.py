from pathlib import Path
from pydantic import BaseModel, BaseSettings
from enum import Enum


class Sex(str, Enum):
    MALE = "M"
    FEMALE = "F"


class PersonalInformation(BaseModel):
    first_name: str
    last_name: str
    birthdate: str
    sex: Sex
    expiration_date: str
    identity_num: str


class Settings(BaseSettings):
    model_checkpoint: str = "models/model/"
