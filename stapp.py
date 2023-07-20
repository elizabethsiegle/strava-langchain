# examples for few-shot prompt templating
examples = [
  {
    "training_start_date": "July 2", "marathon_date": "Dec 10", "pd_output": "average distance is 3.73 mi","plan": 
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
...
{marathon_date}: Marathon Day!
"""
  },
  {
    "training_start_date" : "July 19", "marathon_date" : "Nov 12", "pd_output": "average distance is 3.73 miles", "plan": 
"""
Week 1: 
Day 1, {training_start_date}: Cross-training - Swim
Day 2: Easy run - 4 miles at an easy pace
Day 3: Cross-training - Elliptical 
Day 4: Medium run - 5 miles at a medium pace 
Day 5: Cross-training - Weight Training
Day 6: Long run - 6 miles at an easy pace 
Day 7: Rest day 
...
{marathon_date}: Marathon Day!
"""
  }
]