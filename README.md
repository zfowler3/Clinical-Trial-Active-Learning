# Clinical Trial Active Learning

To run prospective active learning, type in a terminal window:
```
run-retro-prosp-learning.py --visit_mode=yes
```

To run retrospective active learning, type in a terminal window:
```
run-retro-prosp-learning.py
```

Other parameters like query size ( --nquery ) can be modified. Can set fixed test size via --forgetting_mode='fixed' or dynamic test size via --forgetting_mode='fixed'. --dynamic_test_size is by default 0 when using the dynamic test set (indicates adding in entire visit at each round)
