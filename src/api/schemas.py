"""Request and response contracts for the PAD screening API."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field

SexLevel = Literal["Male", "Female"]
RaceLevel = Literal["Mexican American", "Other Hispanic", "Non-Hispanic White",
               "Non-Hispanic Black", "Other/Multi-racial"]
SmokingLevel = Literal["Never", "Former", "Current"]
DiabetesLevel = Literal["No", "Borderline", "Yes"]
YesNoLevel = Literal["No", "Yes"]


class PatientInput(BaseModel):
    """One patient's clinical record, as typed into the form.

    Ranges mirror the observed spread of the training data. Fields that are
    nullable in the training set stay optional here - the pipelines impute them.
    """
    Age: float = Field(ge=18, le=85, description="Years")
    Sex: SexLevel
    Race: RaceLevel
    BMI: float = Field(ge=14, le=60, description="kg/m^2")
    Smoking: Optional[SmokingLevel] = None
    Diabetes: Optional[DiabetesLevel] = None
    HbA1c: Optional[float] = Field(default=None, ge=3.5, le=19, description="%")
    SBP: float = Field(ge=70, le=240, description="Systolic BP, mmHg")
    DBP: float = Field(ge=30, le=130, description="Diastolic BP, mmHg")
    Total_Cholesterol: float = Field(ge=80, le=500, description="mg/dL")
    HDL: float = Field(ge=20, le=160, description="mg/dL")
    Medical_History_Hypertension: Optional[YesNoLevel] = None

    # Measured, never modelled: used only to grade severity.
    ABI: float = Field(ge=0.0, le=3.0, description="Ankle-brachial index (measured)")

    @computed_field
    @property
    def TC_HDL_Ratio(self) -> float:
        """Derived column in the training data, so derive it here too."""
        return round(self.Total_Cholesterol / self.HDL, 2)

    def to_features(self) -> dict:
        """Model feature dict - ABI is deliberately excluded."""
        data = self.model_dump()
        data.pop("ABI")
        return data


class ABIAssessment(BaseModel):
    value: float
    band: str
    label: str
    detail: str


class PredictionResponse(BaseModel):
    pad_detected: bool
    verdict: str
    confidence: float = Field(description="0.5-1.0 confidence in the verdict returned")
    probability: float = Field(description="Raw ensemble probability of PAD")
    risk: ABIAssessment
    threshold: float
