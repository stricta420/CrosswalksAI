class DecisionMaker:
    def __init__(self):
        pass

        def make_decision(self, analysis_result,n):
            disble_with_age, ages = analysis_result

            if len(disble_with_age) > 0:
                max_age = max(disble_with_age)
            elif len(ages) > 0:
                max_age = max(ages)
            else:
                return "No people detected"

            if max_age <= 30:
                n = 1*n
            elif 30 < max_age <= 50:
                n = 2*n
            elif 50 < max_age <= 65:
                n = 3*n
            else:
                n = 4*n

            return n
