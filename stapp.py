# examples for few-shot prompt templating
examples = [
  {
    "training_start_date": "July 2",  
    "marathon_date": "Dec 10",
    "pd_output": "average distance is 3.73 mi",
    "plan": 
"""
Good luck on {marathon_date}. 
Week 1: 
Day 1, {training_start_date}: tennis for 1 hour
Day 2: 4 miles easy run 
Day 3: 3 miles easy run 
Day 4: 5 miles easy run 
Day 5: bike for 45 minutes
Day 6: 6 miles easy run 
Day 7: 4 miles easy run 
Week 2: 
Day 8: swim for 30 minutes 
Day 9: 4 miles easy run 
Day 10: 3 miles easy run 
Day 11: 6 miles easy run 
Day 12: tennis for 1.5 hours 
Day 13: 7 miles easy run 
Day 14: 4 miles easy run 
Week 3: 
Day 15: tennis for an hour 
Day 16: 5 miles easy run 
Day 17: 4 miles easy run 
Day 18: 7 miles easy run 
Day 19: swim for 30 minutes 
Day 20: 8 miles easy run 
Day 21: 4 miles easy run 
Week 4: 
Day 22: bike 8 miles 
Day 23: 5 miles easy run 
Day 24: 4 miles easy run 
Day 25: 8 miles easy run 
Day 26: bike 10 miles 
Day 27: 9 miles easy run 
Day 28: 4 miles easy run 
Week 5:
Day 29: tennis for an hour and a half 
Day 30: 6 miles easy run 
Day 31: 4 miles easy run 
Day 32: 9 miles easy run 
Day 33: tennis match
Day 34: 10 miles easy run 
Day 35: 4 miles easy run 
Week 6: 
Day 36: bike ride to and from Golden Gate Park (8 miles) 
Day 37: 6 miles easy run 
Day 38: 4 miles easy run 
Day 39: 10 miles easy run 
Day 40: tennis for an hour 
Day 41: 11 miles easy run 
Day 42: 4 miles easy run 
Week 7: 
Day 43: tennis for an hour and a half 
Day 44: 7 miles easy run 
Day 45: 4 miles easy run 
Day 46: 11 miles easy run 
Day 47: tennis for an hour 
Day 48: 12 miles easy run 
Day 49: 4 miles easy run 
Week 8: 
Day 50: swim for 30 minutes 
Day 51: 7 miles easy run 
Day 52: 4 miles easy run 
Day 53: 12 miles easy run 
Day 54: tennis for 1.5 hours 
Day 55: 13 miles easy run 
Day 56: 4 miles easy run 
Week 9: 
Day 57: swim for 30 minutes 
Day 58: 8 miles easy run 
Day 59: 4 miles easy run 
Day 60: 13 miles easy run 
Day 61: bike for 1 hour 
Day 62: 14 miles easy run 
Day 63: 4 miles easy run
Week 10: 
Day 64: swim for 30 minutes  
Day 65: 8 miles easy run 
Day 66: 4 miles easy run 
Day 67: 14 miles easy run 
Day 68: tennis for 1 hour 
Day 69: 15 miles easy run 
Day 70: 4 miles easy run 
Week 11: 
Day 71: tennis for 1 hour 
Day 72: 9 miles easy run 
Day 73: 4 miles easy run 
Day 74: 15 miles easy run 
Day 75: tennis for 1 hour 
Day 76: 16 miles easy run 
Day 77: 4 miles easy run 
Week 12: 
Day 78: swim for 30 minutes 
Day 79: 9 miles easy run 
Day 80: 4 miles easy run 
Day 81: 16 miles easy run 
Day 82: swim for 30 minutes 
Day 83: 17 miles easy run 
Day 84: 4 miles easy run 
Week 13: 
Day 85: tennis for 1 hour 
Day 86: 10 miles easy run 
Day 87: 4 miles easy run 
Day 88: 17 miles easy run 
Day 89: bike ride for 45 minutes 
Day 90: 18 miles easy run
Day 91: 4 miles easy run 
Week 14: 
Day 92: tennis for 1 hour 
Day 93: 10 miles easy run 
Day 94: 4 miles easy run 
Day 95: 18 miles easy run 
Day 96: swim for 30 minutes 
Day 97: 19 miles easy run 
Day 98: 4 miles easy run 
Week 15: 
Day 99: tennis for 1 hour 
Day 100: 11 miles easy run 
Day 101: 4 miles easy run 
Day 102: 19 miles easy run 
Day 103: tennis for 1 hour 
Day 104: 20 miles easy run 
Day 105: 4 miles easy run 
Week 16: 
Day 106: swim for 30 minutes 
Day 107: 11 miles easy run 
Day 108: 4 miles easy run 
Day 109: 20 miles easy run 
Day 110: tennis for 1 hour 
Day 111: 10 miles easy run 
Day 112: 4 miles easy run 
Week 17: 
Day 113: tennis for 1.5 hours 
Day 114: 10 miles easy run 
Day 115: 4 miles easy run 
Day 116: 10 miles easy run 
Day 117: bike ride for 45 minutes 
Day 118: 8 miles easy run 
Day 119: 4 miles easy run 
Week 18: 
Day 120: tennis for 1 hour 
Day 121: 8 miles easy run 
Day 122: 4 miles easy run 
Day 123: 8 miles easy run 
Day 124: tennis for 1 hour 
Day 125: 6 miles easy run 
Day 126: 4 miles easy run 
Week 19: 
Day 127: tennis for 1 hour 
Day 128: 6 miles easy run 
Day 129: 4 miles easy run 
Day 130: 6 miles easy run 
Day 131: tennis for 1 hour 
Day 132: 4 miles easy run 
Day 133: 4 miles easy run 
Week 20: 
Day 134: swim for 20 minutes 
Day 135: 4 miles easy run 
Day 136: 4 miles easy run 
Day 137: 4 miles easy run 
Day 138: bike for 30 minutes 
Day 139: 3 miles easy run 
Day 140: 4 miles easy run 
Week 21: 
Day 141: tennis for 1 hour 
Day 142: 3 miles easy run 
Day 143: 4 miles easy run 
Day 144: 3 miles easy run 
Day 145: swim for 20 minutes 
Day 146: 2 miles easy run 
Day 147: 4 miles easy run 
Week 22: 
Day 148: bike for 30 minutes 
Day 149: 2 miles easy run 
Day 150: 4 miles easy run 
Day 151: 2 miles easy run 
Day 152: swim for 20 minutes
Day 153: 1 mile easy run 
Day 154: 4 miles easy run 
Week 23: 
Day 155: tennis for 1 hour 
Day 156: 1 mile easy run 
Day 157: 4 miles easy run 
Day 158: 1 mile easy run 
Day 159: Rest 
Day 160: Rest 
{marathon_date}: Marathon Day!
"""
  },
  {
    "training_start_date" : "July 19",
    "marathon_date" : "Nov 12",
    "pd_output": "average distance is 3.73 miles",
    "plan": 
"""
Week 1: 
Day 1, {training_start_date}: Cross-training - Swim
Day 2: Easy run - 4 miles at an easy pace
Day 3: Cross-training - Elliptical 
Day 4: Medium run - 5 miles at a medium pace 
Day 5: Cross-training - Weight Training
Day 6: Long run - 6 miles at an easy pace 
Day 7: Rest day 
Week 2: 
Day 8: Easy run - 4 miles at an easy pace
Day 9: Cross-training - Ride 
Day 10: Medium run - 5 miles at a medium pace 
Day 11:Cross-training - Swim 
Day 12: Long run - 7 miles at an easy pace 
Day 13: Rest day 
Day 14: Sprint workout - 3 miles at a hard pace 
Week 3: 
Day 15: Easy run - 4 miles at an easy pace 
Day 16: Cross-training - Bike ride 
Day 17: Medium run - 6 miles at a medium pace 
Day 18: Cross-training - Elliptical 
Day 19: Long run - 8 miles at an easy pace 
Day 20: Rest day 
Day 21: Sprint workout - 3 miles at a hard pace 
Week 4: 
Day 22: Easy run - 5 miles at an easy pace 
Day 23: Cross-training - Ride
Day 24: Medium run - 6 miles at a medium pace 
Day 25: Cross-training - Weight Training 
Day 26: Long run - 9 miles at an easy pace 
Day 27: Rest day 
Day 28: Sprint workout - 4 miles at a hard pace
Week 5: 
Day 29: Easy run - 5 miles at an easy pace 
Day 30: Cross-training - Swim 
Day 31: Medium run - 7 miles at a medium pace 
Day 32: Cross-training - Walk 
Day 33: Long run - 10 miles at an easy pace 
Day 34: Rest day 
Day 35: Sprint workout - 4 miles at a hard pace 
Week 6: 
Day 36: Easy run - 5 miles at an easy pace 
Day 37: Cross-training - Elliptical 
Day 38: Medium run - 7 miles at a medium pace 
Day 39: Cross-training - Ride 
Day 40: Long run - 11 miles at an easy pace 
Day 41: Rest day 
Day 42: Sprint workout - 5 miles at a hard pace 
Week 7: 
Day 43: Easy run - 6 miles at an easy pace
Day 44: Cross-training - Weight Training 
Day 45: Medium run - 8 miles at a medium pace 
Day 46: Cross-training - Swim 
Day 47: Long run - 12 miles at an easy pace 
Day 48: Rest day 
Day 49: Sprint workout - 5 miles at a hard pace 
Week 8: 
Day 50: Easy run - 6 miles at an easy pace 
Day 51: Cross-training - Walk 
Day 52: Medium run - 8 miles at a medium pace 
Day 53: Cross-training - Elliptical 
Day 54: Long run - 13 miles at an easy pace 
Day 55: Rest day 
Day 56: Sprint workout - 6 miles at a hard pace 
Week 9: 
Day 57: Easy run - 6 miles at an easy pace 
Day 58: Cross-training - Ride 
Day 59: Medium run - 9 miles at a medium pace 
Day 60: Cross-training - Weight Training 
Day 61: Long run - 14 miles at an easy pace 
Day 62: Rest day 
Day 63: Sprint workout - 6 miles at a hard pace
Week 10: 
Day 64: Easy run - 7 miles at an easy pace 
Day 65: Cross-training - Swim 
Day 66: Medium run - 9 miles at a medium pace 
Day 67: Cross-training - Walk 
Day 68: Long run - 14 miles at an easy pace 
Day 69: Rest day 
Day 70: Sprint workout - intervals 
Week 11: 
Day 71: Easy run - 7 miles at an easy pace 
Day 72: Cross-training - Elliptical and Weight Training
Day 73: Medium run - 10 miles at a medium pace 
Day 74: Cross-training - Ride 
Day 75: Long run - 14 miles at an easy pace 
Day 76: Rest day 
Day 77: Sprint workout - intervals
Week 12: 
Day 78: Easy run - 7 miles at an easy pace
Day 79: Cross-training - Elliptical and Weight Training 
Day 80: Medium run - 10 miles at a medium pace 
Day 81: Cross-training - Swim 
Day 82: Long run - 15 miles at an easy pace 
Day 83: Rest day 
Day 84: Sprint workout - intervals 
Week 13: 
Day 85: Easy run - 8 miles at an easy pace 
Day 86: Cross-training - Walk 
Day 87: Medium run - 11 miles at a medium pace 
Day 88: Cross-training - Elliptical and Weight Training
Day 89: Long run - 16 miles at an easy pace 
Day 90: Rest day 
Day 91: Sprint workout - 1 mile at a hard pace 
Week 14: 
Day 92: Easy run - 8 miles at an easy pace 
Day 93: Cross-training - Ride 
Day 94: Medium run - 11 miles at a medium pace 
Day 95: Cross-training - Weight Training 
Day 96: Long run - 17 miles at an easy pace 
Day 97: Rest day 
Day 98: Sprint workout - 1 mile at a hard pace
Week 15: 
Day 99: Easy run - 8 miles at an easy pace 
Day 100: Cross-training - Swim 
Day 101: Medium run - 10 miles at a medium pace 
Day 102: Cross-training - bike ride 
Day 103: Long run - 18 miles at an easy pace 
Day 104: Rest day 
Day 105: Sprint workout intervals 
Week 16:
Day 106: Easy run - 8 miles at an easy pace  
Day 107: Cross-training - Tennis
Day 108: Medium run - 11 miles at a medium pace
Day 109: Cross-training - Elliptical and Weight Training
Day 110: Long run - 12 miles at an easy pace 
Day 111: Rest day 
Day 112: Sprint workout - 3 miles at a hard pace
{marathon_date}: Marathon Day!
Note: The above plan assumes that the student is currently able to comfortably run 4 miles at an easy pace. Adjustments may need
to be made if the student is not yet at this level. Additionally, the plan includes cross-training and rest days to
ensure proper recovery and prevent injury.
"""
  }
]

