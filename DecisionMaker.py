import math
class DecisionMaker:
    def __init__(self):
        pass

    # def make_decision(self, analysis_result,n):
    #     disble_with_age, ages = analysis_result

    #     if len(disble_with_age) > 0:
    #         max_age = max(disble_with_age)
    #     elif len(ages) > 0:
    #         max_age = max(ages)
    #     else:
    #         return 0

    #     if max_age <= 30:
    #         n = 1*n
    #     elif 30 < max_age <= 50:
    #         n = 2*n
    #     elif 50 < max_age <= 65:
    #         n = 3*n
    #     else:
    #         n = 4*n

    #     return n
    
    def make_decision(self, analysis_result, n):
        disble_with_age, ages = analysis_result

        disability_max = 0
        people_amount = len(ages)
        max_age = 0
        age_mult = 0
        pmult=0

        if len(disble_with_age) > 0:
            for dis in disble_with_age:
                if dis.label == "Crutches" and disability_max<1:
                    disability_max=1
                if dis.label == "Wheelchair" and disability_max<3:
                    disability_max=3
                if dis.label == "Push_wheelchair" and disability_max<2:
                    disability_max=2
                if dis.label == "Walking_frame" and disability_max<4:
                    disability_max=4
        if len(ages) > 0:
            max_age = max(ages)

        if people_amount>1:
            pmult = math.log(people_amount)
        
        return round((n+disability_max*2+max_age//15)*(1+pmult))
