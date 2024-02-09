- What would a lack of software-only singularity look like in this framework??
    - Run out of good ideas and have to keep waiting for experiments?
- Related to above, what important differences are there between this model and the real world? What is meant by "ideas getting harder to find" anyway?
    - It takes longer to brainstorm new ideas because the old ones are gone?
    - Claim is that it takes 2x cumulative effort to get ~5x AE. How can we model that?
    - Maybe the idea distribution should be even more skewed? idk
- Should I take into account limited compute for speedups and/or running experiments?
    - Maybe each experiment should require compute and if you use up your compute budget you can't start another experiment?
    - Max experiments at once as a hacky way?
- Fiddle with AE distribution again
- Use human range intuition for what effects of taste should be
- Run multiple sims with the same variable values
- Run MC sims with varying variable values (e.g. MC, sensitivity analyses)
- Make FLOP automation schedue for various tasks readable from a spreadsheet a la Epoch (and ideally other variables)
- Try to roughly back out r and see if it seems reasonable?
    - Think about how to measure cog output. Prob not num experiments done?
- Nicer output: log to csv, graph, etc.
- Add command line options
- Introduce correlation for lengths besides runtime
- Maybe some experiments should be locked by other experiments? Or a dumb way to model it would be that finishing experiments increases taste?



misc:
Constraints to satisfy:
- Possible to not do SOS
- Gives significant progress next few years (even with no automation)
- Gets intuition behind diminishing software returns
- Total amount of effective compute from ideas reasonable?
