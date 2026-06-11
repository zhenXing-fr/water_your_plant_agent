# src/garden_agent/domain/models.py

from enum import Enum
from datetime import date
from pydantic import BaseModel, Field

class GrowthStage(str, Enum):
    SEEDLING = "seedling"
    ESTABLISHED = "established"
    MATURE = "mature"

class SoilType(str, Enum):
    SANDY = "sandy"
    CLAY = "clay"
    LOAMY = "loamy"
    WELL_DRAINING = "well_draining"

class Plant(BaseModel):
    name: str
    plant_type: str                    # "tomato", "succulent", "lavender"
    growth_stage: GrowthStage
    soil_type: SoilType
    last_watered: date | None = None
    notes: str = ""

class Garden(BaseModel):
    id: str
    location: str                      # used by weather adapter for geocoding
    plants: list[Plant]

class WeatherForecast(BaseModel):
    date: date
    temperature_max_c: float
    temperature_min_c: float
    precipitation_mm: float
    humidity_percent: float
    is_rain_expected: bool

class WateringAction(BaseModel):
    plant_name: str
    amount_liters: float
    time_of_day: str                   # "morning" or "evening"
    reason: str                        # why — forces LLM to explain itself

class DailyPlan(BaseModel):
    date: date
    actions: list[WateringAction]
    skip_reason: str | None = None     # e.g. "rain forecast > 10mm"

class WateringPlan(BaseModel):
    garden_id: str
    week_start: date
    daily_plans: list[DailyPlan]