Make usable:
- Make automation schedues readable from a spreadsheet a la Epoch (and ideally other important variables)
- Add command line options
- Log to CSV

More robust results:
- Run multiple sims with the same variable values
- Run MC sims with varying variable values (e.g. MC, sensitivity analyses)

Tune params:
- Should num experiments allowed go up as AE increases?
- Shoule there be more AE available?
- Should results somehow be more consistnt?
- Use human range intuition for what effects of taste should be
- Fit no automation case to expections (e.g. r) more thoroughly
        - Claim is that it takes 2x cumulative effort to get ~5x AE. How to model?
- take initial distribution -> run with no automation for a while, take the ones left for our starting point?? Not sure this makes sense

Nice to haves:
- Introduce correlation for lengths besides runtime
- Research teams have different tastes / capabilities? 

Fundamental qs:
- What would a lack of software-only singularity look like in this framework??
    - Run out of good ideas and have to keep waiting for experiments?
- Related to above, what important differences are there between this model and the real world? What is meant by "ideas getting harder to find" anyway?
    - It takes longer to brainstorm new ideas because the old ones are gone? meh
    - Maybe some experiments should be locked by other experiments? Or a dumb way to model it would be that finishing experiments increases taste?

Bug fixes:
- Fix diminishing dip when experiments run out
