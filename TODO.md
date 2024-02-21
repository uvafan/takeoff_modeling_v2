TODO first:
- Change EC/AE to t-AGI stuff, matching the spreadsheet model? (maybe leave EC/AE as an option)
    - Could also change to human range based on AI Impacts thing?
- Incorporate alg eff thinking updates

Fundamental qs:
- Figure out whether wall clock time should be incorporated in true priority?
- Adjust runtime requirements over time -> increased num experiments?
- The assumption that EM doesn’t have to be parallelized is “conservative” in that there are likely serial bottlenecks from EM as well, which aren’t modeled
-  Simulate much better taste by adding in new good experiments? To account for there being more than 10k experiments in the real world??
- Incorporate inference compute bottlenecks for speedups?
    - Cost for ai speedup. Or at least have some botecs on why the automation schedule is consistent with it
- Should we add pre-training runs as a bottleneck to improved alg efficiency?
    - Maybe distinguish betwen alg efficiency for each category as was originally proposed? idk
- What would a lack of software-only singularity look like in this framework??
    - Run out of good ideas and have to keep waiting for experiments?
- Related to above, what important differences are there between this model and the real world? What is meant by "ideas getting harder to find" anyway?
    - It takes longer to brainstorm new ideas because the old ones are gone? meh
    - Maybe some experiments should be locked by other experiments? Or a dumb way to model it would be that finishing experiments increases taste?

Make usable:
- Make automation schedues readable from a spreadsheet a la Epoch (and ideally other important variables). Also standardize them to be easier (1ex, 1ex+1, etc.)
- Write GDoc explainer
- Add command line options
- Log to CSV
- Finish other lognormals

More robust results:
- Run multiple sims with the same variable values
- Run MC sims with varying variable values (e.g. MC, sensitivity analyses)

Improve parameter choices:
- Smarter true priority that takes into account runtime length...
- Should num experiments go up as physical compute increases?
    - e.g. maybe the biggest model scales faster than the compute needed for "informative small-scale experiments". Does this theory backtest well?
- Shoule there be more AE available?
- Use human range intuition for what effects of taste should be
- Fit no automation case to expections (e.g. r) more thoroughly
    - Claim is that it takes 2x cumulative effort to get ~5x AE. How to model?
- Should results somehow be more consistent?
- take initial distribution -> run with no automation for a while, take the ones left for our starting point?? Not sure this makes sense

Nice to haves:
- Introduce correlation for lengths besides runtime.
- Research teams have different tastes / capabilities? 
- Look into issue where thingy goes down at the end
- Be able to switch immediately when an experiment stops away from the serial setup, if that’s advantageous
Or choose in a smarter way when there are multiple stopped experiments?

Bug fixes:
- Fix diminishing dip when experiments run out
