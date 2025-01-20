class DecisionMaker:
    def __init__(self):
        pass

    def make_decision(self, analysis_result):
        disble_with_age, ages = analysis_result
        # Example decision logic
        if len(ages) > 0:
            max_age = max(ages)
            if max_age > 60:
                return "Caution: Elderly person detected"
            elif len(disble_with_age) > 0:
                return "Caution: Disabled person detected"
            else:
                return "All clear"
        else:
            return "No people detected"
