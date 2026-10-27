"""Target routing only; never routes policy images or other observation fields."""
def target_appearance(observed_appearance,mode='appearance_specific'):
 assert observed_appearance in ('clean','background')
 assert mode in ('appearance_specific','shared_clean')
 return 'clean' if mode=='shared_clean' else observed_appearance
