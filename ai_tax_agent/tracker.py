class TaxSimplificationState:
    def __init__(self, target_length: int, target_dollars: int):
        self.prior_version = 0
        self.working_version = 1
        self.prior_character_length = 0
        self.current_character_length = 0
        self.target_length = target_length
        self.target_dollars = target_dollars # Represents the initial or target total impact
        self.remaining_length_to_process = 0
        self.kept_dollars = 0 # Total dollars from sections KEPT
        self.removed_dollars = 0 # Total dollars from sections REMOVED
        # self.remaining_dollars = target_dollars # Removed this line as target_dollars serves this purpose

    def clear(self, target_length: int = None, target_dollars: int = None):
        """Resets the state for a new run."""
        self.prior_version = 0
        self.working_version = 1
        self.prior_character_length = 0
        self.current_character_length = 0
        if target_length is not None:
            self.target_length = target_length
        if target_dollars is not None: # Allow resetting target dollars
             self.target_dollars = target_dollars
        self.remaining_length_to_process = 0
        self.kept_dollars = 0 # Reset kept dollars
        self.removed_dollars = 0 # Reset removed dollars
        # self.remaining_dollars = self.target_dollars # Reset remaining to target (removed)

    def increment_current_length(self, text: str):
        """Adds the length of processed text to the current working version."""
        self.current_character_length += len(text)

    def set_prior_state(self, prior_version: int, prior_character_length: int, remaining_length_to_process: int):
        """Explicitly set prior state values."""
        self.prior_version = prior_version
        self.working_version = prior_version + 1
        self.prior_character_length = prior_character_length
        self.remaining_length_to_process = remaining_length_to_process

    def decrement_remaining_length(self, text: str):
        """Subtracts the processed section length from remaining length to process."""
        self.remaining_length_to_process -= len(text)
        if self.remaining_length_to_process < 0:
            self.remaining_length_to_process = 0  # Safety: No negative lengths

    def track_kept_dollars(self, amount: int):
        """Adds the specified amount to the total dollars kept."""
        if amount > 0:
            self.kept_dollars += amount

    def track_removed_dollars(self, amount: int):
        """Adds the specified amount to the total dollars removed."""
        if amount > 0:
            self.removed_dollars += amount

    def summary(self) -> dict:
        """Returns a summary of the current state as a dictionary."""
        return {
            "prior_version": self.prior_version,
            "working_version": self.working_version,
            "prior_character_length": self.prior_character_length,
            "current_character_length": self.current_character_length,
            "target_length": self.target_length,
            "remaining_length_to_process": self.remaining_length_to_process,
            "target_dollars": self.target_dollars,
            "kept_dollars": self.kept_dollars,
            "removed_dollars": self.removed_dollars,
        }

