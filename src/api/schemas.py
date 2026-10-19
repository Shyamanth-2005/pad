"""Request and response contracts for the PAD screening API."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field, model_validator

SexLevel = Literal["Male", "Female"]
RaceLevel = Literal["Mexican American", "Other Hispanic", "Non-Hispanic White",
               "Non-Hispanic Black", "Other/Multi-racial"]
SmokingLevel = Literal["Never", "Former", "Current"]
DiabetesLevel = Literal["No", "Borderline", "Yes"]
YesNoLevel = Literal["No", "Yes"]


class AccelerometryInput(BaseModel):
    """Seven-day hip accelerometer summaries (report Table 3.2).

    Each value is averaged over valid days: at least four days with 10+ hours of
    wear. Bounds are physical limits of a 1,440-minute day and ActiGraph counts.
    """
    Log_Total_Counts: float = Field(ge=0, le=18, description="log(1 + counts/day)")
    Sedentary_Share: float = Field(ge=0, le=1, description="Share of wear time < 100 counts/min")
    Light_Minutes: float = Field(ge=0, le=1440, description="Minutes/day at 100-2019 counts/min")
    MVPA_Minutes: float = Field(ge=0, le=1440, description="Minutes/day at >= 2020 counts/min")
    MVPA_Bout_Minutes: float = Field(ge=0, le=1440, description="MVPA minutes/day in bouts >= 10 min")
    ASTP: float = Field(ge=0, le=1, description="Active-to-sedentary transition probability")
    SATP: float = Field(ge=0, le=1, description="Sedentary-to-active transition probability")
    Peak30_Counts: float = Field(ge=0, le=32767, description="Mean of the 30 most active minutes")
    Longest_Active_Bout: float = Field(ge=0, le=1440, description="Minutes, >= 100 counts/min")
    Log_PM_AM_Ratio: float = Field(ge=-18, le=18, description="log afternoon / morning counts")

    @model_validator(mode="after")
    def bouts_within_mvpa(self):
        if self.MVPA_Bout_Minutes > self.MVPA_Minutes:
            raise ValueError("MVPA_Bout_Minutes cannot exceed MVPA_Minutes")
        return self


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

    # Optional sensor branch; present only when a valid recording exists (m = 1).
    # Accepted across the full 18-85 range, although the sensor model was fitted
    # on NHANES 2003-2004, which measured the ABI only from age 40.
    Accelerometry: Optional[AccelerometryInput] = None

    @computed_field
    @property
    def TC_HDL_Ratio(self) -> float:
        """Derived column in the training data, so derive it here too."""
        return round(self.Total_Cholesterol / self.HDL, 2)

    def to_features(self) -> dict:
        """Clinical feature dict - ABI and the sensor summaries are excluded."""
        return self.model_dump(exclude={"ABI", "Accelerometry"})


class ABIAssessment(BaseModel):
    value: float
    band: str
    label: str
    detail: str


class PredictionResponse(BaseModel):
    pad_detected: bool
    verdict: str
    confidence: float = Field(description="0.5-1.0 confidence in the verdict returned")
    probability: float = Field(description="Probability of PAD the verdict is based on: "
                                           "the fusion layer's score, with the sensor "
                                           "branch when accelerometry is supplied")
    risk: ABIAssessment
    threshold: float
    modalities: list[str] = Field(description="Branches that contributed to the verdict")
    branch_probabilities: dict[str, float] = Field(
        description="Per-branch probability of PAD before fusion: clinical "
                    "(augmented-trained ensemble), nhanes (real-cohort model), "
                    "sensor (when supplied)")
