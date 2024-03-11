# Learning Fortran by Teaching Fortran to Learn
This is my most recent project. It is intended to teach myself Fortran, with an eye on maybe running GPT on GPU in the future. I'm trying to follow Andrej Karpathy's tutorial while writing all required functionality as well. (There is even a merge-sort-implementation so I could implement `random_choice` reliably.)

Don't use. Not even close to production ready! Currently breaks valgrind. Even testing is homebrew!

## Curernt state
The code compiles with `gfortran network_utils.f08 network_layers.f08 structured_network.f08 file_helpers.f08 network_training.f08 -Wall` without warnings. This will build an executable from the `program` block in `network_training.f08`, where several different scenarios are trained. First, a network to turn one-hot encoded numbers into binary is trained (and expected to fail). Then a perceptron is trained to approximate a simple parabola. Last, a network is trained on the content of "names.txt" to produce more names.