# Notes

For some reason applying a random 2D rotation seems to make things much harder
- JK actually it's fine
- Ok so it seems like a bad random rotation/dataset sample can mess things up

Going to higher dimensions seems to make things even easier???
- yeah seems to be true. Why?
- doesn't seem to affect the log_linear fit much, but does affect diverging behavior at end of training

Intrinsic dimension definitely seems to have a big effect?

NN architecture has an even bigger effect though
- too small or too big and we diverge

According to initial experiments:
- intrinsic dimension has a SIGNIFICANT effect on convergence rate
- extrinsic dimension much less so

log_linear coefficient appears to depend logarithmically on the intrinsic dimension?

Not sure how much of this is due to the model not really learning anything

Fixing the addition bug doesn't seem to have changed much

The findings also appear to be more or less the same across all batch sizes?

# Resources

Look at NTK maybe? Can they say something about optimiztion with low-dimensional structure in the dataset
- no one can really prove anything for two-layer neural

Haomin paper: paper on understanding why NNs can't achieve machine level accuracy
- gradient descent must solve linear system which is very ill-conditioned. This results in numercal error at 
- <https://arxiv.org/pdf/2306.17301.pdf>

Some initial takeaways:
- log_linear decreases (in absolute value) with d, but not linearly. 
- for very small d (d = 2) embedding in a much higher dimension seems to be good. But for larger d this flips to what we expect (larger D is slightly worse)

# ToDos

- Try simpler model (LDA, logistic)
- Talk to peter bartlett
- extract $\alpha_C$ from optimization pre-convergence (this is what openai does)
- try removing SGD for GD
- make this open-source