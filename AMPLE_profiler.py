import cProfile
from cProfile import Profile
from AMPLE_fn import main

run_time_list = []

with cProfile.Profile() as pr:
    for idx in range(0,3):
        run_time = main()
        run_time_list.append(run_time)
        
pr.dump_stats('AMPLE_prof_3run.prof')

# Then run following commands in Anaconda prompt
# cd C:\Users\ellie\OneDrive\Documents\University\Year 4\Modules\Final Year Project\Code\MY Python vs Matlab\Python AMPLE
# AMPLE_collapse_prof_1run


import cProfile
from cProfile import Profile
from AMPLE_fn import main

with cProfile.Profile() as pr:
        run_time = main()
        
pr.dump_stats('AMPLE_prof.prof')