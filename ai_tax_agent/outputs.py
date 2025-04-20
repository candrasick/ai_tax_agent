#!/usr/bin/env python
"""Defines Pydantic models for expected outputs from agents."""

from typing import Literal, Optional
from pydantic import BaseModel, Field
from decimal import Decimal

# Define Pydantic model for editor output
class TaxEditorOutput(BaseModel):
    """Structure representing the expected JSON output from the Tax Editor Agent."""
    action: Literal["simplify", "redraft", "delete", "keep"] = Field(description="The final decision on how to handle the section.")
    rationale: str = Field(description="The reasoning behind the chosen action.")
    after_text: Optional[str] = Field(default=None, description="The simplified or redrafted text, if applicable.")
    # Use Decimal for currency, allow None
    estimated_deleted_dollars: Optional[Decimal] = Field(default=None, description="Estimated financial impact if the section is deleted.")
    estimated_kept_dollars: Optional[Decimal] = Field(default=None, description="Estimated financial impact if the section is kept/simplified/redrafted.")
    new_complexity_score: Optional[float] = Field(default=None, description="The estimated complexity score of the text after changes (or original if kept).") 